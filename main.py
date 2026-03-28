import sqlite3
import pandas as pd
import requests
import json
import webbrowser
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# ================= CONFIGURATION =================
# Laptop 2: The SQL Specialist
SQL_API = "http://172.168.0.8:1337/v1/chat/completions"
SQL_MODEL = "sqlcoder-7b-q5_k_m.gguf"

# Laptop 1 (Local): The Reasoning & Explanation Brain
QWEN_API = "http://127.0.0.1:1337/v1/chat/completions"
QWEN_MODEL = "Qwen2.5-Coder-7B-Instruct-Q6_K_L.gguf"

DB_NAME = "analytics_platform.db"
CSV_FOLDER = "archive"

# ================= RAG GLOBALS =================
rag_model = None
rag_index = {}  # {table_name: np.ndarray}

# ================= 1. DYNAMIC DATABASE ENGINE =================
def initialize_database(folder_path):
    """Scans folder for CSVs and creates a multi-table SQLite DB."""
    conn = sqlite3.connect(DB_NAME)
    if not os.path.exists(folder_path):
        print(f"❌ Folder '{folder_path}' not found!")
        return

    print("📥 Initializing Database from CSV Repository...")
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            # Clean filename to create a valid SQL table name
            table_name = file.replace(".csv", "").replace("olist_", "").replace("_dataset", "")
            file_path = os.path.join(folder_path, file)

            try:
                df = pd.read_csv(file_path)
                df.to_sql(table_name, conn, if_exists="replace", index=False)
                print(f"  ✅ Loaded '{file}' as table: [{table_name}]")
            except Exception as e:
                print(f"  ⚠️ Failed to load {file}: {e}")
    conn.close()
    print("🚀 All tables synchronized.\n")


def get_full_schema():
    """Extracts schema for ALL tables dynamically."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    schema_text = ""
    for table in tables:
        t_name = table[0]
        schema_text += f"\nTable: {t_name}\nColumns:\n"
        cursor.execute(f"PRAGMA table_info({t_name})")
        for col in cursor.fetchall():
            schema_text += f"  - {col[1]} ({col[2]})\n"
    conn.close()
    return schema_text


def get_multi_table_samples():
    """Gets sample rows from every table for semantic context."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    samples = ""
    for table in tables:
        t_name = table[0]
        cursor.execute(f"SELECT * FROM {t_name} LIMIT 3")
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        samples += f"\n[Table: {t_name}]\nCols: {', '.join(cols)}\n"
        for r in rows:
            samples += f"  Sample: {str(r)}\n"
    conn.close()
    return samples


# ================= 2. RAG ENGINE =================
def build_rag_index():
    """Embed each table's schema + sample data for semantic retrieval."""
    global rag_model, rag_index
    print("🧩 Building RAG index...")
    rag_model = SentenceTransformer("all-MiniLM-L6-v2")

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cursor.fetchall()]

    for t in tables:
        cursor.execute(f"PRAGMA table_info({t})")
        col_info = cursor.fetchall()
        schema_str = f"Table {t} columns: " + ", ".join(f"{c[1]} ({c[2]})" for c in col_info)

        cursor.execute(f"SELECT * FROM {t} LIMIT 3")
        rows = cursor.fetchall()
        col_names = [c[1] for c in col_info]
        sample_str = "Sample rows: " + "; ".join(
            str(dict(zip(col_names, r))) for r in rows
        )

        doc = schema_str + ". " + sample_str
        rag_index[t] = rag_model.encode(doc, convert_to_numpy=True)

    conn.close()
    print(f"✅ RAG index ready: {len(rag_index)} tables embedded.\n")


def retrieve_relevant_tables(question, top_k=4):
    """Return top-k most semantically relevant tables for the question."""
    if not rag_index or rag_model is None:
        return list(rag_index.keys())

    q_vec = rag_model.encode(question, convert_to_numpy=True)
    scores = {
        t: float(np.dot(q_vec, v) / (np.linalg.norm(q_vec) * np.linalg.norm(v) + 1e-9))
        for t, v in rag_index.items()
    }
    top = sorted(scores, key=scores.get, reverse=True)[:top_k]
    print(f"📎 RAG → relevant tables: {top}")
    return top


def get_schema_for_tables(table_names):
    """Build schema text for a specific subset of tables."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    schema_text = ""
    for t in table_names:
        schema_text += f"\nTable: {t}\nColumns:\n"
        cursor.execute(f"PRAGMA table_info({t})")
        for col in cursor.fetchall():
            schema_text += f"  - {col[1]} ({col[2]})\n"
    conn.close()
    return schema_text


def get_samples_for_tables(table_names):
    """Get sample rows only for specified tables."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    samples = ""
    for t in table_names:
        cursor.execute(f"SELECT * FROM {t} LIMIT 3")
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        samples += f"\n[Table: {t}]\nCols: {', '.join(cols)}\n"
        for r in rows:
            samples += f"  Sample: {str(r)}\n"
    conn.close()
    return samples


# ================= 3. INTELLIGENT AGENTS =================
def analyze_intent(question, schema):
    print("🔍 Analyzing Intent...")
    prompt = f"""
    Classify the intent of this database question.
    Schema: {schema}
    Question: {question}

    Intents:
    - visualize: User wants a graph, ER diagram, or visual map.
    - schema: User wants to know the structure/tables/columns in text.
    - semantic: User asks what a column means or represents.
    - data: User wants a specific answer (requires SQL).
    - invalid: Unrelated to this database.

    Output ONLY a JSON: {{"intent": "...", "refined": "clear version of question"}}
    """

    try:
        response = requests.post(QWEN_API, json={
            "model": QWEN_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0
        })
        raw = response.json()['choices'][0]['message']['content'].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except:
        return {"intent": "data", "refined": question}


def build_er_diagram(tables_filter=None):
    """Programmatically builds an accurate Mermaid erDiagram — no LLM hallucination."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    all_tables = [r[0] for r in cursor.fetchall()]
    tables = tables_filter if tables_filter else all_tables

    table_cols = {}
    for t in tables:
        cursor.execute(f"PRAGMA table_info({t})")
        table_cols[t] = [(col[1], col[2].split("(")[0].strip() or "TEXT") for col in cursor.fetchall()]
    conn.close()

    lines = ["erDiagram"]
    for t, cols in table_cols.items():
        lines.append(f"    {t} {{")
        for col_name, col_type in cols:
            lines.append(f"        {col_type} {col_name}")
        lines.append("    }")

    # Detect relationships via shared *_id columns
    tlist = list(table_cols.keys())
    for i, t1 in enumerate(tlist):
        for t2 in tlist[i+1:]:
            shared = {c[0] for c in table_cols[t1]} & {c[0] for c in table_cols[t2]}
            for col in sorted(shared):
                if col.endswith("_id"):
                    lines.append(f'    {t1} ||--o{{ {t2} : "{col}"')

    return "\n".join(lines)


def generate_sql(question, schema):
    print("⚙️ Generating Multi-Table SQL...")
    prompt = f"Schema:\n{schema}\nQuestion: {question}\nRules: SQLite syntax. Use JOINs if needed. ONLY output SQL."
    response = requests.post(SQL_API, json={
        "model": SQL_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0
    }, headers={"Host": "localhost"})
    sql = response.json()['choices'][0]['message']['content']
    return sql.replace("```sql", "").replace("```", "").strip()


# ================= 3. UTILITIES & RENDERING =================
def render_mermaid(code):
    clean_code = code.replace("```mermaid", "").replace("```", "").strip()
    html = f"""<html><body style="background:#1e1e1e; display:flex; justify-content:center; padding:20px;">
        <pre class="mermaid">{clean_code}</pre>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true, theme: 'dark' }});
        </script>
    </body></html>"""

    with open("diagram.html", "w") as f:
        f.write(html)
    webbrowser.open("diagram.html")
    print("🌐 ER Diagram opened in browser.")


def execute_sql(sql):
    """Execute SQL and return results."""
    print(f"\n📋 Executing SQL:\n{sql}\n")
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        conn.close()
        return cols, rows
    except Exception as e:
        conn.close()
        return None, str(e)


def explain_results(question, sql, cols, rows):
    """Use Qwen to explain SQL results in plain English."""
    print("💡 Generating Explanation...")
    sample = str(rows[:5])
    prompt = f"""
    Question: {question}
    SQL Used: {sql}
    Columns: {cols}
    Sample Results: {sample}
    Total Rows: {len(rows)}

    Provide a clear, concise explanation of what the results show. Be specific with numbers.
    """
    try:
        response = requests.post(QWEN_API, json={
            "model": QWEN_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        })
        return response.json()['choices'][0]['message']['content'].strip()
    except:
        return f"Query returned {len(rows)} rows."


# ================= 4. MAIN ORCHESTRATOR =================
def process_question(question):
    schema = get_full_schema()
    intent_data = analyze_intent(question, schema)
    intent = intent_data.get("intent", "data")
    refined = intent_data.get("refined", question)

    print(f"\n🎯 Intent: [{intent.upper()}] | Refined: {refined}\n")

    if intent == "invalid":
        print("❌ This question doesn't relate to the database.")

    elif intent == "visualize":
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        all_tables = [r[0] for r in cursor.fetchall()]
        conn.close()

        # Match table names from original question AND refined (Qwen may drop table names)
        search_text = (question + " " + refined).lower()
        mentioned = [t for t in all_tables if t.lower() in search_text]
        filter_tables = mentioned if len(mentioned) >= 1 else None

        if filter_tables:
            print(f"📊 Building ER diagram for: {', '.join(filter_tables)}")
        else:
            print("📊 Building full ER diagram...")

        diagram_code = build_er_diagram(filter_tables)
        render_mermaid(diagram_code)

    elif intent == "schema":
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        table_names = [r[0] for r in cursor.fetchall()]
        conn.close()
        q = refined.lower()

        if any(w in q for w in ["how many", "count", "number of", "total"]):
            print(f"\n📐 There are {len(table_names)} tables in the database.\n")

        elif any(w in q for w in ["list", "what tables", "which tables", "show tables", "all tables"]):
            print(f"\n📐 Tables ({len(table_names)}):")
            for t in table_names:
                print(f"  - {t}")
            print()

        else:
            # Check if asking about a specific table
            matched = next((t for t in table_names if t.lower() in q), None)
            if matched:
                conn2 = sqlite3.connect(DB_NAME)
                cursor2 = conn2.cursor()
                cursor2.execute(f"PRAGMA table_info({matched})")
                cols = cursor2.fetchall()
                conn2.close()
                print(f"\n📐 Table '{matched}' has {len(cols)} columns:")
                for col in cols:
                    print(f"  - {col[1]} ({col[2]})")
                print()
            else:
                # Full schema fallback
                print(f"\n📐 Database Schema ({len(table_names)} tables):\n")
                print(schema)

    elif intent == "semantic":
        relevant_tables = retrieve_relevant_tables(refined, top_k=3)
        rag_schema = get_schema_for_tables(relevant_tables)
        samples = get_samples_for_tables(relevant_tables)
        prompt = f"Schema:\n{rag_schema}\nSamples:\n{samples}\nQuestion: {refined}\nExplain what this column/table represents."
        try:
            response = requests.post(QWEN_API, json={
                "model": QWEN_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3
            })
            answer = response.json()['choices'][0]['message']['content'].strip()
            print(f"\n💬 Answer:\n{answer}\n")
        except Exception as e:
            print(f"❌ Error: {e}")

    else:  # data intent
        relevant_tables = retrieve_relevant_tables(refined, top_k=4)
        rag_schema = get_schema_for_tables(relevant_tables)
        sql = generate_sql(refined, rag_schema)
        cols, rows = execute_sql(sql)

        if cols is None:
            print(f"❌ SQL Error: {rows}")
        else:
            print(f"✅ {len(rows)} row(s) returned.")
            if rows:
                print("\n" + " | ".join(cols))
                print("-" * 60)
                for row in rows[:10]:
                    print(" | ".join(str(v) for v in row))
                if len(rows) > 10:
                    print(f"... and {len(rows) - 10} more rows.")

            explanation = explain_results(question, sql, cols, rows)
            print(f"\n💡 Insight:\n{explanation}\n")


# ================= 5. MAIN ENTRY POINT =================
if __name__ == "__main__":
    initialize_database(CSV_FOLDER)
    build_rag_index()

    print("=" * 60)
    print("  🧠 Multi-LLM Analytics Platform")
    print("  SQL Brain: SQLCoder-7B | Reasoning: Qwen2.5-Coder-7B")
    print("=" * 60)

    while True:
        question = input("\n❓ Ask a question (or 'quit' to exit): ").strip()
        if question.lower() in ["quit", "exit", "q"]:
            print("👋 Goodbye!")
            break
        if question:
            process_question(question)