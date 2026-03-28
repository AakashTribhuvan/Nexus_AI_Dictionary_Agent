# app.py — Web API for Multi-LLM Analytics Platform
import os
import json
import sqlite3
import numpy as np
import pandas as pd
import requests
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ═══════════════════════════ CONFIGURATION ════════════════════════════
SQL_API    = "http://172.168.0.8:1337/v1/chat/completions"
SQL_MODEL  = "sqlcoder-7b-q5_k_m.gguf"
QWEN_API   = "http://127.0.0.1:1337/v1/chat/completions"
QWEN_MODEL = "Qwen2.5-Coder-7B-Instruct-Q6_K_L.gguf"
DB_NAME    = "analytics_platform.db"
CSV_FOLDER = "archive"

# ═══════════════════════════ RAG GLOBALS ══════════════════════════════
rag_model: SentenceTransformer | None = None
rag_index: dict[str, np.ndarray] = {}

# ═══════════════════════════ DATABASE ENGINE ══════════════════════════
def initialize_database(folder_path: str):
    conn = sqlite3.connect(DB_NAME)
    if not os.path.exists(folder_path):
        print(f"  ⚠  Folder '{folder_path}' not found — skipping CSV import.")
        conn.close()
        return
    print("📥 Initializing database from CSV files...")
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            table = (
                file.replace(".csv", "")
                    .replace("olist_", "")
                    .replace("_dataset", "")
            )
            try:
                df = pd.read_csv(os.path.join(folder_path, file))
                df.to_sql(table, conn, if_exists="replace", index=False)
                print(f"  ✅ '{file}' → [{table}]")
            except Exception as e:
                print(f"  ⚠  Failed to load {file}: {e}")
    conn.close()
    print("🚀 Database ready.\n")


def get_all_tables() -> list[str]:
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in c.fetchall()]
    conn.close()
    return tables


def get_full_schema() -> str:
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in c.fetchall()]
    schema = ""
    for t in tables:
        schema += f"\nTable: {t}\nColumns:\n"
        c.execute(f"PRAGMA table_info({t})")
        for col in c.fetchall():
            schema += f"  - {col[1]} ({col[2]})\n"
    conn.close()
    return schema


def get_schema_for_tables(names: list[str]) -> str:
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    schema = ""
    for t in names:
        schema += f"\nTable: {t}\nColumns:\n"
        c.execute(f"PRAGMA table_info({t})")
        for col in c.fetchall():
            schema += f"  - {col[1]} ({col[2]})\n"
    conn.close()
    return schema


def get_samples_for_tables(names: list[str]) -> str:
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    out = ""
    for t in names:
        c.execute(f"SELECT * FROM {t} LIMIT 3")
        rows = c.fetchall()
        cols = [d[0] for d in c.description]
        out += f"\n[Table: {t}]\nCols: {', '.join(cols)}\n"
        for r in rows:
            out += f"  {str(r)}\n"
    conn.close()
    return out


# ═══════════════════════════ RAG ENGINE ══════════════════════════════
def build_rag_index():
    global rag_model, rag_index
    print("🧩 Building RAG index...")
    rag_model = SentenceTransformer("all-MiniLM-L6-v2")
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in c.fetchall()]
    for t in tables:
        c.execute(f"PRAGMA table_info({t})")
        col_info = c.fetchall()
        schema_str = f"Table {t} columns: " + ", ".join(
            f"{ci[1]} ({ci[2]})" for ci in col_info
        )
        c.execute(f"SELECT * FROM {t} LIMIT 3")
        rows = c.fetchall()
        col_names = [ci[1] for ci in col_info]
        sample_str = "Sample rows: " + "; ".join(
            str(dict(zip(col_names, r))) for r in rows
        )
        rag_index[t] = rag_model.encode(
            schema_str + ". " + sample_str, convert_to_numpy=True
        )
    conn.close()
    print(f"✅ RAG index ready — {len(rag_index)} tables embedded.\n")


def retrieve_relevant_tables(question: str, top_k: int = 4) -> list[str]:
    if not rag_index or rag_model is None:
        return list(rag_index.keys())
    q_vec = rag_model.encode(question, convert_to_numpy=True)
    scores = {
        t: float(np.dot(q_vec, v) / (np.linalg.norm(q_vec) * np.linalg.norm(v) + 1e-9))
        for t, v in rag_index.items()
    }
    return sorted(scores, key=scores.get, reverse=True)[:top_k]


# ═══════════════════════════ AGENTS ══════════════════════════════════
def analyze_intent(question: str, schema: str) -> dict:
    prompt = f"""Classify the intent of this database question.
Schema: {schema}
Question: {question}

Intents:
- visualize: User wants a graph, ER diagram, or visual map.
- schema: User wants to know the structure/tables/columns in text.
- semantic: User asks what a column means or represents.
- data: User wants a specific answer (requires SQL).
- invalid: Unrelated to this database.

Output ONLY a JSON: {{"intent": "...", "refined": "clear version of question"}}"""
    try:
        r = requests.post(QWEN_API, json={
            "model": QWEN_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        })
        raw = r.json()["choices"][0]["message"]["content"].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception:
        return {"intent": "data", "refined": question}


def build_er_diagram(tables_filter: list[str] | None = None) -> str:
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    all_tables = [r[0] for r in c.fetchall()]
    tables = tables_filter or all_tables
    table_cols: dict[str, list[tuple]] = {}
    for t in tables:
        c.execute(f"PRAGMA table_info({t})")
        table_cols[t] = [
            (col[1], col[2].split("(")[0].strip() or "TEXT")
            for col in c.fetchall()
        ]
    conn.close()

    lines = ["erDiagram"]
    for t, cols in table_cols.items():
        lines.append(f"    {t} {{")
        for col_name, col_type in cols:
            lines.append(f"        {col_type} {col_name}")
        lines.append("    }")

    tlist = list(table_cols.keys())
    for i, t1 in enumerate(tlist):
        for t2 in tlist[i + 1 :]:
            shared = {c[0] for c in table_cols[t1]} & {c[0] for c in table_cols[t2]}
            for col in sorted(shared):
                if col.endswith("_id"):
                    lines.append(f'    {t1} ||--o{{ {t2} : "{col}"')
    return "\n".join(lines)


def generate_sql(question: str, schema: str) -> str:
    prompt = (
        f"Schema:\n{schema}\nQuestion: {question}\n"
        "Rules: SQLite syntax. Use JOINs if needed. ONLY output SQL."
    )
    r = requests.post(
        SQL_API,
        json={
            "model": SQL_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        headers={"Host": "localhost"},
    )
    sql = r.json()["choices"][0]["message"]["content"]
    return sql.replace("```sql", "").replace("```", "").strip()


def execute_sql(sql: str) -> tuple:
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        c.execute(sql)
        rows = c.fetchall()
        cols = [d[0] for d in c.description]
        conn.close()
        return cols, rows
    except Exception as e:
        conn.close()
        return None, str(e)


def explain_results(question: str, sql: str, cols: list, rows: list) -> str:
    prompt = f"""Question: {question}
SQL Used: {sql}
Columns: {cols}
Sample Results: {str(rows[:5])}
Total Rows: {len(rows)}
Provide a clear, concise explanation of what the results show. Be specific with numbers."""
    try:
        r = requests.post(QWEN_API, json={
            "model": QWEN_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        })
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return f"Query returned {len(rows)} row(s)."


# ═══════════════════════════ SERIALIZATION ════════════════════════════
def _safe(v):
    if v is None:
        return None
    if isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return str(v)


def serialize_rows(rows: list) -> list[list]:
    return [[_safe(v) for v in row] for row in rows]


# ═══════════════════════════ ORCHESTRATOR ════════════════════════════
def process_question_api(question: str) -> dict:
    schema = get_full_schema()
    intent_data = analyze_intent(question, schema)
    intent  = intent_data.get("intent", "data")
    refined = intent_data.get("refined", question)
    result: dict = {"intent": intent, "refined": refined}

    if intent == "invalid":
        result["message"] = "This question doesn't relate to the database."

    elif intent == "visualize":
        all_tables  = get_all_tables()
        search_text = (question + " " + refined).lower()
        mentioned   = [t for t in all_tables if t.lower() in search_text]
        filter_tables = mentioned or None
        result["filter_tables"] = filter_tables or all_tables
        result["mermaid_code"]  = build_er_diagram(filter_tables)

    elif intent == "schema":
        table_names = get_all_tables()
        q = refined.lower()
        if any(w in q for w in ["how many", "count", "number of", "total"]):
            result["subtype"]     = "count"
            result["table_count"] = len(table_names)
        elif any(w in q for w in ["list", "what tables", "which tables", "show tables", "all tables"]):
            result["subtype"] = "list"
            result["tables"]  = table_names
        else:
            matched = next((t for t in table_names if t.lower() in q), None)
            if matched:
                conn = sqlite3.connect(DB_NAME)
                c = conn.cursor()
                c.execute(f"PRAGMA table_info({matched})")
                cols = c.fetchall()
                c.execute(f"SELECT COUNT(*) FROM {matched}")
                row_count = c.fetchone()[0]
                conn.close()
                result["subtype"]      = "table_detail"
                result["detail_table"] = matched
                result["row_count"]    = row_count
                result["columns"]      = [
                    {"name": col[1], "type": col[2], "pk": bool(col[5])}
                    for col in cols
                ]
            else:
                result["subtype"]     = "full"
                result["schema_text"] = schema
                result["table_count"] = len(table_names)

    elif intent == "semantic":
        rel = retrieve_relevant_tables(refined, top_k=3)
        result["relevant_tables"] = rel
        rag_schema = get_schema_for_tables(rel)
        samples    = get_samples_for_tables(rel)
        prompt = (
            f"Schema:\n{rag_schema}\nSamples:\n{samples}\n"
            f"Question: {refined}\nExplain what this column/table represents."
        )
        try:
            r = requests.post(QWEN_API, json={
                "model": QWEN_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
            })
            result["answer"] = r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            result["error"] = str(e)

    else:  # data
        rel = retrieve_relevant_tables(refined, top_k=4)
        result["relevant_tables"] = rel
        rag_schema = get_schema_for_tables(rel)
        try:
            sql = generate_sql(refined, rag_schema)
            result["sql"] = sql
            cols, rows = execute_sql(sql)
            if cols is None:
                result["error"] = rows  # error string
            else:
                result["columns"]     = cols
                result["rows"]        = serialize_rows(rows[:500])
                result["total_rows"]  = len(rows)
                result["explanation"] = explain_results(question, sql, cols, rows)
        except Exception as e:
            result["error"] = str(e)

    return result


# ═══════════════════════════ FASTAPI APP ══════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_database(CSV_FOLDER)
    build_rag_index()
    yield


app = FastAPI(title="Analytics Platform", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


@app.post("/api/ask")
async def api_ask(req: AskRequest):
    try:
        return JSONResponse(process_question_api(req.question))
    except Exception as e:
        return JSONResponse({"intent": "error", "error": str(e)}, status_code=500)


@app.get("/api/tables")
async def api_tables():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in c.fetchall()]
    info = []
    for t in tables:
        c.execute(f"SELECT COUNT(*) FROM {t}")
        rows = c.fetchone()[0]
        c.execute(f"PRAGMA table_info({t})")
        cols = len(c.fetchall())
        info.append({"name": t, "rows": rows, "columns": cols})
    conn.close()
    return {"tables": info}


@app.get("/api/schema/{table_name}")
async def api_schema(table_name: str):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(f"PRAGMA table_info({table_name})")
    cols = [
        {"name": col[1], "type": col[2], "notnull": bool(col[3]), "pk": bool(col[5])}
        for col in c.fetchall()
    ]
    c.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = c.fetchone()[0]
    conn.close()
    return {"table": table_name, "columns": cols, "row_count": count}


@app.get("/api/status")
async def api_status():
    return {
        "status": "ready",
        "db": DB_NAME,
        "tables": len(get_all_tables()),
        "rag_tables": len(rag_index),
    }


os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)