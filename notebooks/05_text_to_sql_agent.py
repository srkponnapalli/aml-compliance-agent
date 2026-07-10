# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Text-to-SQL Agent
# MAGIC
# MAGIC Lets an analyst ask ad-hoc questions in natural language instead of relying on the
# MAGIC hardcoded queries in notebook 03. The flow:
# MAGIC
# MAGIC 1. **Introspect** the `northstar.compliance` schema from Unity Catalog `information_schema`
# MAGIC 2. **Generate** a SQL query from the question + schema context (Llama 3.3 70B)
# MAGIC 3. **Guard** the query — SELECT/WITH only; any DML/DDL is rejected before execution
# MAGIC 4. **Execute** it with `spark.sql`
# MAGIC 5. **Explain** the results back in natural language
# MAGIC
# MAGIC Requires the tables from notebook `02_synthetic_data` to exist.

# COMMAND ----------

import re
import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")

CATALOG = "northstar"
SCHEMA = "compliance"
GEN_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Schema introspection
# MAGIC
# MAGIC Pull column names and types straight from Unity Catalog so the prompt always reflects the
# MAGIC live schema. A few categorical sample values are added to help the model use exact strings
# MAGIC (e.g. `'Cash Deposit'`, `'High'`).

# COMMAND ----------

def get_schema_context(catalog=CATALOG, schema=SCHEMA):
    cols = spark.sql(f"""
        SELECT table_name, column_name, data_type, ordinal_position
        FROM {catalog}.information_schema.columns
        WHERE table_schema = '{schema}'
        ORDER BY table_name, ordinal_position
    """).collect()

    tables = {}
    for row in cols:
        tables.setdefault(row["table_name"], []).append(f"{row['column_name']} {row['data_type']}")

    lines = []
    for table, columns in tables.items():
        lines.append(f"Table {catalog}.{schema}.{table}:")
        for col in columns:
            lines.append(f"  - {col}")
        lines.append("")

    # A few distinct categorical values help the model match exact strings.
    hints = [
        "Known categorical values:",
        "  transactions.transaction_type: 'Cash Deposit', 'Cash Withdrawal', 'Wire Transfer', 'ACH Transfer', 'Check Deposit'",
        "  transactions.channel: 'Branch', 'ATM', 'Online', 'Mobile'",
        "  customers.risk_rating: 'Low', 'Medium', 'High'",
        "  sanctioned_countries.severity: 'Comprehensive', 'Sectoral'",
    ]
    return "\n".join(lines + hints)


SCHEMA_CONTEXT = get_schema_context()
print(SCHEMA_CONTEXT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Natural language → SQL

# COMMAND ----------

SQL_SYSTEM_PROMPT = """You are a Text-to-SQL assistant for an AML compliance analyst working on \
Databricks (Spark SQL / Unity Catalog).

Rules:
- Return exactly ONE SQL statement and nothing else. No explanation, no commentary.
- Generate a read-only SELECT query only. Never write INSERT, UPDATE, DELETE, MERGE, DROP,
  ALTER, CREATE, TRUNCATE, or GRANT.
- Use fully qualified table names (catalog.schema.table).
- Use only the tables and columns provided in the schema.
- Prefer explicit JOINs on the documented keys (customer_id, destination_country -> country).
- Wrap the query in a single ```sql code block."""


def extract_sql(text):
    """Pull the SQL out of a fenced code block, or fall back to the raw text."""
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    sql = match.group(1) if match else text
    return sql.strip().rstrip(";").strip()


def text_to_sql(question, schema_context=SCHEMA_CONTEXT):
    messages = [
        {"role": "system", "content": SQL_SYSTEM_PROMPT},
        {"role": "user", "content": f"Schema:\n{schema_context}\n\nQuestion:\n{question}"},
    ]
    response = client.predict(endpoint=GEN_ENDPOINT, inputs={"messages": messages})
    return extract_sql(response["choices"][0]["message"]["content"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Safety guard
# MAGIC
# MAGIC The LLM is instructed to emit read-only SQL, but we never trust that — this is the hard
# MAGIC gate. Only a single `SELECT`/`WITH` statement is allowed to reach `spark.sql`.

# COMMAND ----------

FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|MERGE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|REPLACE|COPY|REFRESH)\b",
    re.IGNORECASE,
)


def is_safe_select(sql):
    stripped = sql.strip().rstrip(";").strip()
    if ";" in stripped:  # reject stacked/multiple statements
        return False, "Multiple statements are not allowed."
    if not re.match(r"^(SELECT|WITH)\b", stripped, re.IGNORECASE):
        return False, "Only SELECT/WITH queries are allowed."
    if FORBIDDEN.search(stripped):
        return False, "Query contains a forbidden (write/DDL) keyword."
    return True, "ok"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Execute

# COMMAND ----------

def run_query(sql):
    safe, reason = is_safe_select(sql)
    if not safe:
        raise ValueError(f"Refused to run query: {reason}\n\nSQL:\n{sql}")
    return spark.sql(sql)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Natural language answer
# MAGIC
# MAGIC End-to-end: question → SQL → results → plain-English answer. Returns the SQL, the result
# MAGIC DataFrame, and the narrative so the analyst can audit the generated query.

# COMMAND ----------

ANSWER_SYSTEM_PROMPT = "You are an AML compliance assistant for NorthStar Financial Services. \
Answer the analyst's question using only the query results provided. Be concise and specific; \
cite customer names, amounts, and transaction IDs where relevant."


def answer_question(question, max_rows=50, verbose=True):
    sql = text_to_sql(question)

    safe, reason = is_safe_select(sql)
    if not safe:
        return {"question": question, "sql": sql, "error": reason, "answer": None, "dataframe": None}

    df = spark.sql(sql)
    pdf = df.limit(max_rows).toPandas()
    results_text = pdf.to_string(index=False) if not pdf.empty else "(no rows returned)"

    messages = [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question:\n{question}\n\nSQL used:\n{sql}\n\nResults:\n{results_text}"},
    ]
    response = client.predict(endpoint=GEN_ENDPOINT, inputs={"messages": messages})
    answer = response["choices"][0]["message"]["content"]

    if verbose:
        print(f"Q: {question}\n")
        print(f"SQL:\n{sql}\n")
        print(f"Answer:\n{answer}")

    return {"question": question, "sql": sql, "error": None, "answer": answer, "dataframe": df}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Demo
# MAGIC
# MAGIC Skipped when this notebook is pulled in via `%run` (i.e. `RUNNING_AS_MODULE` is set).

# COMMAND ----------

if not globals().get("RUNNING_AS_MODULE", False):
    _ = answer_question("Which high-risk customers made more than one cash deposit in June 2024?")
    _ = answer_question("What is the total dollar value of wire transfers to sanctioned countries, broken down by country?")
    _ = answer_question("Who are the top 5 customers by total transaction amount, and what are their risk ratings?")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Guard check
# MAGIC
# MAGIC Confirms the safety gate blocks a write attempt even if a prompt tries to smuggle one in.

# COMMAND ----------

if not globals().get("RUNNING_AS_MODULE", False):
    print(is_safe_select("SELECT * FROM northstar.compliance.customers"))
    print(is_safe_select("DROP TABLE northstar.compliance.customers"))
    print(is_safe_select("SELECT 1; DELETE FROM northstar.compliance.customers"))
