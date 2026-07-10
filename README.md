
# aml-compliance-agent

![Databricks](https://img.shields.io/badge/Databricks-Mosaic%20AI-FF3621?style=flat&logo=databricks&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)
![Delta Lake](https://img.shields.io/badge/Delta%20Lake-Unity%20Catalog-00ADD8?style=flat)
![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%26%20Deployment-0194E2?style=flat&logo=mlflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

A financial compliance agent built natively on Databricks that detects transaction violations and explains them against regulatory policy documents — combining SQL-based transaction monitoring with Retrieval-Augmented Generation (RAG) over compliance policy text.


---

## What It Does

Most compliance tools tell you *what* is wrong. This agent tells you *what*, *why*, and *where in the policy it says so* — in a single natural language response.

**Example output:**

```
User: "Are there any compliance violations in our recent transactions?"

Agent: After reviewing recent transactions, I identified three potential violations:

1. CTR Violation: James Whitfield deposited $12,500 in cash (T0001). Per Policy
   Clause 2.1.1, any cash transaction exceeding $10,000 must be reported via CTR
   within 15 calendar days.

2. Structuring Violation: Arash Mohammadi (High-risk) made 4 cash deposits totalling
   $38,400 between June 3–9, each just under $10,000. This matches structuring
   behavior prohibited under 31 U.S.C. 5324 and Policy Clause 2.2.1.

3. OFAC Violation: Linda Chen wired $15,000 to Iran (T0006). Per Policy Clause 5.2.1,
   transactions involving comprehensively sanctioned jurisdictions must be immediately
   blocked without a valid OFAC license.
```

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│              Compliance Agent               │
│         (databricks-meta-llama-3.3-70b)     │
└──────────────┬──────────────────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌─────────────┐  ┌─────────────────┐
│  RAG Tool   │  │   SQL Tool      │
│             │  │                 │
│ PDF Policy  │  │ Delta Lake      │
│ → Chunks    │  │ Transactions    │
│ → Embeddings│  │ Customers       │
│ → Retrieval │  │ Sanctioned      │
│ → Generate  │  │ Countries       │
└─────────────┘  └─────────────────┘
       │                │
       ▼                ▼
  Policy Clause    Violation Data
       │                │
       └───────┬────────┘
               ▼
        Combined Answer
```

### Two-Tool Design

| Tool | Trigger | Data Source |
|------|---------|-------------|
| `search_compliance_policy` | Policy questions, rule lookups, threshold queries | NorthStar Compliance Policy PDF (RAG) |
| `detect_transaction_violations` | Suspicious activity queries, violation detection | Delta Lake — `northstar.compliance` |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Platform | Databricks (Mosaic AI) |
| Embedding Model | `databricks-bge-large-en` (1024-dim) |
| Generation Model | `databricks-meta-llama-3-3-70b-instruct` |
| Vector Store | In-memory Python dict (custom cosine similarity) |
| Transaction Data | Delta Lake — Unity Catalog (`northstar.compliance`) |
| Model Client | `mlflow.deployments` |
| PDF Parsing | `pypdf` |
| Data | Fully synthetic |

---

## Project Structure

```
aml-compliance-agent/
│
├── notebooks/                       # Databricks source-format notebooks (.py)
│   ├── 01_rag_pipeline.py           # PDF ingestion, chunking, embeddings, retrieval
│   ├── 02_synthetic_data.py         # Delta Lake table creation with planted violations
│   ├── 03_sql_agent.py              # SQL violation detection queries
│   ├── 04_compliance_agent.py       # Full agent wiring and demo (%run 01 + 03)
│   └── 05_text_to_sql_agent.py      # Natural-language → SQL over the compliance tables
│
├── data/
│   ├── README.md                    # Where to place the policy PDF
│   └── NorthStar_Compliance_Policy_Manual.pdf   # Synthetic policy document
│
├── scripts/
│   └── northstar_synthetic_data.py  # Standalone, job-runnable data generation script
│
├── requirements.txt                 # Python dependencies
└── README.md
```

> Notebooks are stored in Databricks **source format** (`.py` with `# COMMAND ----------`
> cell markers) for clean git diffs. Import them into a Databricks workspace as notebooks,
> or open them directly with Databricks Repos / the VS Code extension.

### Run order

1. `notebooks/02_synthetic_data.py` — build the `northstar.compliance` Delta tables
2. `notebooks/04_compliance_agent.py` — runs `01_rag_pipeline` and `03_sql_agent` via `%run`, then the demo

---

## RAG Pipeline

Built from scratch without LangChain or vector database dependencies.

### 1. Chunking

```python
def split_text(text, chunk_size=500, overlap=100):
    text = clean_text(text)
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        pos = text.rfind(".", i, i + chunk_size)
        chunk = text[i:pos]
        chunks.append(chunk.strip())
    return chunks
```

- **Chunk size:** 500 characters
- **Overlap:** 100 characters (20%) — preserves sentence boundary context
- **Result:** 44 chunks from the NorthStar policy document

### 2. Embeddings

```python
import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")

response = client.predict(
    endpoint="databricks-bge-large-en",
    inputs={"input": chunks}
)
embeddings = [item['embedding'] for item in response['data']]
```

- Model: `databricks-bge-large-en`
- Dimensions: 1024
- All 44 chunks embedded in a single API call

### 3. In-Memory Vector Store

```python
documents = {}
for i in range(len(embeddings)):
    documents[i] = {"text": chunks[i], "embedding": embeddings[i]}
```

Dict-of-dicts keyed by integer ID — O(1) lookup by ID at retrieval time.

### 4. Cosine Similarity (from scratch)

```python
import math

def dot_product(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))

def magnitude(v):
    return math.sqrt(dot_product(v, v))

def cosine_similarity(a, b):
    return dot_product(a, b) / (magnitude(a) * magnitude(b))
```

### 5. Retrieval

```python
def retrieve(query, documents, top_k=3):
    query_embedding = client.predict(
        endpoint="databricks-bge-large-en",
        inputs={"input": query}
    )['data'][0]['embedding']

    scores = {}
    for i in documents:
        scores[i] = {
            'similarity': cosine_similarity(documents[i]['embedding'], query_embedding),
            'text': documents[i]['text']
        }

    sorted_scores = sorted(scores.items(), key=lambda x: x[1]['similarity'], reverse=True)
    return [score[1]['text'] for score in sorted_scores[:top_k]]
```

### 6. Generation

```python
def generate(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
    messages = [
        {"role": "system", "content": "You are a compliance assistant for NorthStar Financial Services."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
    ]
    response = client.predict(
        endpoint="databricks-meta-llama-3-3-70b-instruct",
        inputs={"messages": messages}
    )
    return response['choices'][0]['message']['content']
```

---

## Text-to-SQL Agent

`05_text_to_sql_agent.py` lets an analyst ask ad-hoc questions in natural language instead of relying on the hardcoded queries in notebook 03.

```
Question ──▶ schema context ──▶ LLM ──▶ SQL ──▶ safety guard ──▶ spark.sql ──▶ LLM ──▶ answer
             (information_schema)                 (SELECT-only)                   (explain)
```

1. **Schema introspection** — columns and types are pulled live from Unity Catalog `information_schema`, plus a few categorical value hints (e.g. `'Cash Deposit'`, `'High'`) so the model uses exact strings.
2. **Generation** — Llama 3.3 70B turns the question + schema into a single SQL statement.
3. **Safety guard** — `is_safe_select()` is the hard gate: only a single `SELECT`/`WITH` statement reaches Spark. Any DML/DDL keyword or stacked statement is rejected before execution — the LLM's instructions are never trusted on their own.
4. **Answer** — results are fed back to the model for a concise natural-language response. The generated SQL is always returned so the analyst can audit it.

```python
result = answer_question("Which high-risk customers made more than one cash deposit in June 2024?")
result["sql"]        # the generated (audited) query
result["dataframe"]  # the Spark DataFrame of results
result["answer"]     # the natural-language explanation
```

---

## Synthetic Data

All data is fully synthetic. Three violations are planted in 10,000 transactions across 100 customers.

### Planted Violations

| # | Type | Customer | Description | Policy Clause |
|---|------|----------|-------------|---------------|
| 1 | CTR | James Whitfield | Single cash deposit of $12,500 | Policy Clause 2.1.1 |
| 2 | Structuring / SAR | Arash Mohammadi (High-risk) | 4 cash deposits of $9,200–$9,900 within 7 days | Policy Clause 2.2.1 |
| 3 | OFAC | Linda Chen | Wire transfer of $15,000 to Iran | Policy Clause 5.2.1 |

### Schema

```
northstar.compliance.customers
northstar.compliance.transactions
northstar.compliance.sanctioned_countries
```

---

## Detection Queries

### CTR Violation
```sql
SELECT t.transaction_id, c.name, c.risk_rating, t.amount, t.transaction_type, t.transaction_date
FROM northstar.compliance.transactions t
JOIN northstar.compliance.customers c ON t.customer_id = c.customer_id
WHERE t.transaction_type = 'Cash Deposit'
AND t.amount > 10000
```

### Structuring / SAR
```sql
SELECT c.name, c.risk_rating, COUNT(*) as num_deposits,
       SUM(t.amount) as total_amount,
       MIN(t.transaction_date) as first_txn,
       MAX(t.transaction_date) as last_txn
FROM northstar.compliance.transactions t
JOIN northstar.compliance.customers c ON t.customer_id = c.customer_id
WHERE t.transaction_type = 'Cash Deposit'
AND t.amount BETWEEN 9000 AND 9999
AND c.risk_rating = 'High'
AND t.transaction_date BETWEEN '2024-06-03' AND '2024-06-10'
GROUP BY c.name, c.risk_rating
HAVING COUNT(*) >= 3
```

### OFAC Sanctions
```sql
SELECT t.transaction_id, c.name, t.amount, t.destination_country,
       t.counterparty_name, s.sanction_program, s.severity
FROM northstar.compliance.transactions t
JOIN northstar.compliance.customers c ON t.customer_id = c.customer_id
JOIN northstar.compliance.sanctioned_countries s ON t.destination_country = s.country
WHERE t.transaction_type = 'Wire Transfer'
```

---

## Limitations & Future Work

- **In-memory vector store** — rebuilt on every cluster restart; production version would use Databricks Vector Search
- **Text-to-SQL** — `05_text_to_sql_agent.py` lets analysts ask ad-hoc questions in natural language; a future step is validating generated SQL against a query allow-list beyond the current read-only guard
- **3 violation types** — extensible to hundreds of rules via a proper TMS (Transaction Monitoring System) integration
- **No authentication layer** — production deployment would use Databricks Apps with workspace SSO

---

## Author

**Siva Ponnapalli** — Data Engineer  
[LinkedIn](https://linkedin.com/in/ponnasivark) · [GitHub](https://github.com/srkponnapalli)

---

## License

MIT
