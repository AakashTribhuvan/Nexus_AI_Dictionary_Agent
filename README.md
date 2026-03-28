# 🧠 AI Data Analyst (Distributed NLIDB System)

THIS IS A MULTI AI MODEL BASED PLATFORM NEEDS TO BE RUN ON MULTIPLE PC's IF A SINGLE PC CANNOT HANDLE THE FILES.
A DEMO VIDEO HAS BEEN ATTACHED FOR THOSE WHO CANNOT RUN IT.

Link To Older Versions ==> https://github.com/AakashTribhuvan/Nexus
**Link To Demo Video ==> https://youtu.be/xXVAllaNMm8**
A multi-agent AI system that converts natural language into database insights using distributed models.

---

## 🚀 Overview

This project is a **Natural Language Interface to Databases (NLIDB)** powered by multiple AI agents.

It allows users to:

* Ask questions in plain English
* Automatically generate SQL queries
* Execute them on a database
* Get clean results
* Receive intelligent explanations

---

## 🏗️ Architecture

```
User
 ↓
Web UI (index.html)
 ↓
app.py (API Layer)
 ↓
main.py (AI Pipeline)
 ↓
SQLite Database
 ↓
AI Explanation
 ↓
User
```

### 🧠 AI Model Roles

| Model    | Responsibility                                            |
| -------- | --------------------------------------------------------- |
| Qwen     | Reasoning, intent classification, validation, explanation |
| SQLCoder | SQL generation                                            |

---

## ⚙️ Features

* 🧠 Natural language → SQL
* 🔍 Intent classification (schema / data / semantic / invalid)
* ⚙️ SQL generation using dedicated model
* 🧪 SQL validation before execution
* 🔧 Auto-repair of broken queries
* 📊 Clean table output
* 💬 Human-like explanations
* 🌐 Web interface

---

## 📁 Project Structure

```
.
├── app.py              # Flask API / server
├── main.py             # Core AI pipeline
├── static/
│   └── index.html      # Frontend UI
```

---

## 📦 Dataset Setup

You need to download the dataset before running the project.

### Steps:

1. Download dataset from Kaggle:
   👉 <link>

2. Extract the files

3. Place the extracted files in the **root directory**:

```
.
├── app.py
├── main.py
├── archive/   <-- place dataset here
```

---

## 🛠️ Installation

```bash
pip install pandas sqlite3 requests flask graphviz
```

---

## ▶️ Running the Project

### 1. Start your local AI servers

* Start **Qwen model**
* Start **SQLCoder model**

Make sure APIs are available at:

```
http://127.0.0.1:1337
http://<your-sqlcoder-ip>:1337
```

---

### 2. Run backend

```bash
python app.py
```

---

### 3. Open frontend

Open in browser:

```
static/index.html
```

---

## 💡 Example Queries

* `How many rows are in the dataset?`
* `What are the columns in the table?`
* `Show first 5 rows`
* `Which payment type is most used?`
* `What does payment_type mean?`

---

## 🧠 How It Works

### 1. Intent Detection

Classifies user query into:

* schema
* data
* semantic
* invalid

---

### 2. Query Refinement

Messy input → structured question

---

### 3. SQL Generation

SQLCoder generates SQLite query

---

### 4. Validation

Qwen verifies correctness

---

### 5. Execution

Runs query on SQLite DB

---

### 6. Explanation

Qwen explains results in plain English

---

## 🔥 What Makes This Special

* Multi-agent architecture
* Distributed AI models
* Self-healing queries
* Semantic understanding (beyond SQL)
* Clean separation of reasoning vs execution

---

## 🏆 One-Line Pitch

> “A multi-agent AI data analyst that understands, queries, validates, and explains data autonomously.”

---

## 🚀 Future Improvements

* 📊 Data visualizations (charts)
* 🧠 Memory-based conversations
* 🔗 Multi-table joins + ER diagrams
* ⚡ Faster inference pipeline

---

## 👨‍💻 Authors

Built as part of a hackathon project.

---

## ⚠️ Notes

* Ensure both AI models are running before starting
* Dataset must be correctly placed
* Works best on local network setup for distributed models

---

## ⭐ If You Like This Project

Give it a ⭐ and build something even crazier 🚀
