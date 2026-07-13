# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Compliance Agent
# MAGIC
# MAGIC Wires three tools into a single agent and lets the LLM pick which one to use per question:
# MAGIC
# MAGIC | Tool | Purpose | Backed by |
# MAGIC |------|---------|-----------|
# MAGIC | `violation_sweep` | The fixed AML sweep (CTR / structuring / OFAC) explained against policy | notebook 03 + RAG |
# MAGIC | `text_to_sql` | Ad-hoc questions answered by generated SQL | notebook 05 |
# MAGIC | `policy_lookup` | Pure policy / rule questions | notebook 01 (RAG) |
# MAGIC
# MAGIC A small **router** LLM call classifies the question, then the agent dispatches to that tool.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the tools
# MAGIC
# MAGIC `%run` executes each child notebook top-to-bottom and pulls its functions into this scope:
# MAGIC `retrieve()`, `generate()`, `documents` (01); `detect_violations()` (03);
# MAGIC `answer_question()` (05). Run notebook `02_synthetic_data` first so the tables exist.
# MAGIC
# MAGIC Setting `RUNNING_AS_MODULE = True` before the `%run`s tells each child to skip its own demo
# MAGIC cells, so only this notebook's demo runs.

# COMMAND ----------

# This flag is read by the child notebooks (via globals()) to skip their demo cells under %run.
RUNNING_AS_MODULE = True

# COMMAND ----------

# MAGIC %run ./01_rag_pipeline

# COMMAND ----------

# MAGIC %run ./03_sql_agent

# COMMAND ----------

# MAGIC %run ./05_text_to_sql_agent

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tools
# MAGIC
# MAGIC Each tool is a plain function taking the user's question and returning a string answer.

# COMMAND ----------

def run_violation_sweep(query):
    """The fixed AML sweep: run the three detection queries, attach the relevant policy
    clause to each finding via RAG, and explain them in natural language."""
    violations = detect_violations()

    violation_context = ""
    for violation_type, data in violations.items():
        if data.strip():  # only include if results exist
            retrieved = retrieve(f"policy rules for {violation_type}", documents, top_k=3)
            policy_context = "\n".join(retrieved)
            violation_context += (
                f"\n\n{violation_type.upper()} FINDINGS:\n{data}"
                f"\n\nRELEVANT POLICY:\n{policy_context}"
            )

    answer = generate(query, [violation_context])
    return answer["choices"][0]["message"]["content"]


def run_text_to_sql(query):
    """Ad-hoc question → generated (guarded) SQL → natural-language answer (notebook 05)."""
    result = answer_question(query, verbose=False)
    if result["error"]:
        return f"Could not answer via SQL: {result['error']}"
    return result["answer"]


def run_policy_lookup(query):
    """Pure policy/rule question answered from the policy document via RAG (notebook 01)."""
    retrieved = retrieve(query, documents, top_k=3)
    answer = generate(query, retrieved)
    return answer["choices"][0]["message"]["content"]


# Backwards-compatible alias for the original entry point.
compliance_agent = run_violation_sweep

# COMMAND ----------

# MAGIC %md
# MAGIC ## Router
# MAGIC
# MAGIC A dedicated LLM call whose only job is to classify the question into one tool name. It's
# MAGIC constrained to reply with a single token, and we defensively fall back to `violation_sweep`
# MAGIC if the reply doesn't match a known tool.

# COMMAND ----------

TOOLS = {
    "violation_sweep": run_violation_sweep,
    "text_to_sql": run_text_to_sql,
    "policy_lookup": run_policy_lookup,
}

ROUTER_PROMPT = """You are a router for an AML compliance assistant. Choose the single best tool \
for the user's question and reply with ONLY the tool name — no punctuation, no explanation.

Tools:
- violation_sweep: the user wants to review or detect suspicious activity / compliance violations
  in the transaction data (e.g. "are there any violations?", "show me structuring", "any OFAC hits?").
- text_to_sql: the user asks a specific, ad-hoc question about the transaction/customer data that
  needs a custom query (e.g. "how many wires over $10k went to Germany in May?", "top 5 customers
  by deposit total?").
- policy_lookup: the user asks about a rule, threshold, or definition from the compliance policy,
  not the data (e.g. "what is the CTR threshold?", "when must a SAR be filed?").

Reply with exactly one of: violation_sweep, text_to_sql, policy_lookup"""


def route_tool(query):
    messages = [
        {"role": "system", "content": ROUTER_PROMPT},
        {"role": "user", "content": query},
    ]
    response = client.predict(
        endpoint="databricks-meta-llama-3-3-70b-instruct",
        inputs={"messages": messages},
    )
    raw = response["choices"][0]["message"]["content"].strip().lower()

    # Robust match: find whichever known tool name appears in the reply.
    for name in TOOLS:
        if name in raw:
            return name
    return "violation_sweep"  # safe default

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agent
# MAGIC
# MAGIC Route → dispatch → answer. Returns the chosen tool alongside the answer so you can see
# MAGIC (and audit) how the question was handled.

# COMMAND ----------

def agent(query, verbose=True):
    tool = route_tool(query)
    answer = TOOLS[tool](query)
    if verbose:
        print(f"[tool: {tool}]\n")
        print(answer)
    return {"tool": tool, "answer": answer}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Demo — one question per route

# COMMAND ----------

# Routes to violation_sweep
_ = agent("Are there any compliance violations in our recent transactions?")

# COMMAND ----------

# Routes to text_to_sql
_ = agent("How many wire transfers were sent to each destination country?")

# COMMAND ----------

# Routes to policy_lookup
_ = agent("What is the threshold for filing a Currency Transaction Report?")
