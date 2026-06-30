# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — SQL Agent
# MAGIC
# MAGIC Hardcoded violation-detection queries against the `northstar.compliance` Delta tables.
# MAGIC Defines `detect_violations()`, which the compliance agent (notebook 04) pulls in via
# MAGIC `%run ./03_sql_agent`.
# MAGIC
# MAGIC Requires the tables from notebook `02_synthetic_data` to exist.

# COMMAND ----------

def detect_violations():
    """Run the three detection queries and return their results as printable strings."""
    results = {}

    # Query 1: CTR violations — cash deposits over the $10,000 reporting threshold
    ctr = spark.sql("""
        SELECT t.transaction_id, c.name, c.risk_rating, t.amount, t.transaction_type, t.transaction_date
        FROM northstar.compliance.transactions t
        JOIN northstar.compliance.customers c ON t.customer_id = c.customer_id
        WHERE t.transaction_type = 'Cash Deposit'
        AND t.amount > 10000
        ORDER BY t.amount DESC
    """)

    # Query 2: Structuring — high-risk customers making 3+ sub-$10K deposits within 7 days
    structuring = spark.sql("""
        SELECT c.name, c.risk_rating,
               COUNT(*) as num_deposits,
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
    """)

    # Query 3: OFAC — wire transfers to sanctioned countries
    ofac = spark.sql("""
        SELECT t.transaction_id, c.name, t.amount, t.destination_country,
               t.counterparty_name, s.sanction_program, s.severity
        FROM northstar.compliance.transactions t
        JOIN northstar.compliance.customers c ON t.customer_id = c.customer_id
        JOIN northstar.compliance.sanctioned_countries s ON t.destination_country = s.country
        WHERE t.transaction_type = 'Wire Transfer'
    """)

    results["ctr"] = ctr.toPandas().to_string(index=True)
    results["structuring"] = structuring.toPandas().to_string(index=True)
    results["ofac"] = ofac.toPandas().to_string(index=True)

    return results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Smoke test

# COMMAND ----------

violations = detect_violations()
for violation_type, data in violations.items():
    print(f"\n── {violation_type.upper()} ──")
    print(data)
