# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Compliance Agent
# MAGIC
# MAGIC Wires the RAG tool (notebook 01) and the SQL detection tool (notebook 03) into a single
# MAGIC agent. For each violation type found in the Delta tables, it retrieves the relevant policy
# MAGIC clause and asks the LLM to explain *what*, *why*, and *where in the policy it says so*.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the RAG pipeline and SQL agent
# MAGIC
# MAGIC `%run` imports `documents`, `retrieve()`, `generate()` (from 01) and `detect_violations()`
# MAGIC (from 03) into this notebook's scope. Run notebook `02_synthetic_data` first so the tables
# MAGIC exist.

# COMMAND ----------

# MAGIC %run ./01_rag_pipeline

# COMMAND ----------

# MAGIC %run ./03_sql_agent

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agent

# COMMAND ----------

def compliance_agent(query):
    # Step 1: Detect violations from Delta Lake
    violations = detect_violations()

    # Step 2: Build context from both SQL results and RAG-retrieved policy
    violation_context = ""
    for violation_type, data in violations.items():
        if data.strip():  # only include if results exist
            retrieved = retrieve(f"policy rules for {violation_type}", documents, top_k=3)
            policy_context = "\n".join(retrieved)
            violation_context += (
                f"\n\n{violation_type.upper()} FINDINGS:\n{data}"
                f"\n\nRELEVANT POLICY:\n{policy_context}"
            )

    # Step 3: Generate the final natural-language answer
    answer = generate(query, [violation_context])
    return answer["choices"][0]["message"]["content"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Demo

# COMMAND ----------

print(compliance_agent("Are there any compliance violations in our recent transactions?"))
