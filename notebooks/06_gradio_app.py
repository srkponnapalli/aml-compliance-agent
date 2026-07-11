# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Gradio Demo App
# MAGIC
# MAGIC A lightweight UI over the compliance agent, for showing the demo without living in notebook
# MAGIC cells. It loads the three tools (notebooks 01/03/05), routes each question with the same
# MAGIC LLM router as notebook 04, and shows **which tool was chosen** alongside the answer — so the
# MAGIC audience sees it routing, not just chatting.
# MAGIC
# MAGIC **Must run inside a notebook on the cluster** — the tools call `spark.sql()`, which only
# MAGIC exists here (not in Model Serving). Run notebook `02_synthetic_data` first so the tables exist.

# COMMAND ----------

# MAGIC %pip install gradio pypdf "typing_extensions>=4.12.0"

# COMMAND ----------

# This notebook is the entry point, so restart unconditionally after the install.
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the tools
# MAGIC
# MAGIC `RUNNING_AS_MODULE` tells the child notebooks to skip their demo cells (and skip their own
# MAGIC restart, since we just restarted above).

# COMMAND ----------

RUNNING_AS_MODULE = True

# COMMAND ----------

# MAGIC %run ./01_rag_pipeline

# COMMAND ----------

# MAGIC %run ./03_sql_agent

# COMMAND ----------

# MAGIC %run ./05_text_to_sql_agent

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agent (mirrors notebook 04)
# MAGIC
# MAGIC Same three tools + LLM router as `04_compliance_agent`, redefined here so the app is
# MAGIC self-contained. If you change the routing logic, update it in both places.

# COMMAND ----------

def run_violation_sweep(query):
    violations = detect_violations()
    violation_context = ""
    for violation_type, data in violations.items():
        if data.strip():
            retrieved = retrieve(f"policy rules for {violation_type}", documents, top_k=3)
            policy_context = "\n".join(retrieved)
            violation_context += (
                f"\n\n{violation_type.upper()} FINDINGS:\n{data}"
                f"\n\nRELEVANT POLICY:\n{policy_context}"
            )
    answer = generate(query, [violation_context])
    return answer["choices"][0]["message"]["content"]


def run_text_to_sql(query):
    result = answer_question(query, verbose=False)
    if result["error"]:
        return f"Could not answer via SQL: {result['error']}"
    return result["answer"]


def run_policy_lookup(query):
    retrieved = retrieve(query, documents, top_k=3)
    answer = generate(query, retrieved)
    return answer["choices"][0]["message"]["content"]


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
    for name in TOOLS:
        if name in raw:
            return name
    return "violation_sweep"


def agent(query):
    tool = route_tool(query)
    return tool, TOOLS[tool](query)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gradio UI

# COMMAND ----------

import gradio as gr

TOOL_LABELS = {
    "violation_sweep": "🔍 Violation Sweep",
    "text_to_sql": "🧮 Text-to-SQL",
    "policy_lookup": "📖 Policy Lookup",
}

# What each route did — this is what makes "fixed vs generated-on-the-fly" explicit.
ROUTE_DETAIL = {
    "violation_sweep": "Ran the **fixed, pre-written** detection queries (CTR / structuring / OFAC) "
                       "and attached the matching policy clause via RAG.",
    "text_to_sql": "Wrote a **custom SQL query on the fly** from your question (shown below).",
    "policy_lookup": "Answered from the **policy manual via RAG** — no database query.",
}

# Examples grouped by which tool they exercise, so the routing is obvious in the demo.
FIXED_EXAMPLES = [
    "Are there any compliance violations in our recent transactions?",
    "Show me any structuring activity.",
]
SQL_EXAMPLES = [
    "What's the average deposit amount for high-risk customers versus low-risk customers?",
    "How many wire transfers went to each country, and which of those countries are sanctioned?",
    "Who are the top 5 customers by total transaction amount?",
    "Which customers made transactions in more than three different channels?",
]
POLICY_EXAMPLES = [
    "How long do we have to file a Currency Transaction Report?",
    "When is a Suspicious Activity Report required?",
]


def respond(question):
    if not question or not question.strip():
        return "", "", "Please enter a question."

    tool = route_tool(question)
    generated_sql = ""
    if tool == "text_to_sql":
        result = answer_question(question, verbose=False)
        answer = result["answer"] or f"Could not answer: {result['error']}"
        generated_sql = result["sql"] or ""
    else:
        answer = TOOLS[tool](question)

    detail = ROUTE_DETAIL.get(tool, "")
    if generated_sql:
        detail += f"\n\n**Generated SQL:**\n```sql\n{generated_sql}\n```"
    return TOOL_LABELS.get(tool, tool), detail, answer


with gr.Blocks(title="NorthStar Compliance Agent") as demo:
    gr.Markdown(
        "# NorthStar Compliance Agent\n"
        "Ask a question. The agent **routes** it to one of three tools — watch which one it picks:\n"
        "- 🔍 **Violation Sweep** — *fixed* AML checks (CTR / structuring / OFAC) + policy\n"
        "- 🧮 **Text-to-SQL** — writes a *custom query* for ad-hoc questions\n"
        "- 📖 **Policy Lookup** — answers rules/thresholds from the policy manual (RAG)"
    )
    question = gr.Textbox(label="Question", placeholder="e.g. Are there any compliance violations?", lines=2)
    ask = gr.Button("Ask", variant="primary")
    tool_used = gr.Textbox(label="Router chose", interactive=False)
    detail = gr.Markdown()
    answer = gr.Markdown()

    ask.click(respond, inputs=question, outputs=[tool_used, detail, answer])
    question.submit(respond, inputs=question, outputs=[tool_used, detail, answer])

    gr.Markdown("### Try these — notice how the router sends each group to a different tool")
    gr.Markdown("**Fixed AML sweep** — pre-written checks")
    gr.Examples(FIXED_EXAMPLES, inputs=question, label="Fixed sweep")
    gr.Markdown("**Ad-hoc questions** — SQL generated live")
    gr.Examples(SQL_EXAMPLES, inputs=question, label="Ad-hoc / Text-to-SQL")
    gr.Markdown("**Policy questions** — RAG over the manual")
    gr.Examples(POLICY_EXAMPLES, inputs=question, label="Policy")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Launch
# MAGIC
# MAGIC `share=True` asks Gradio for a public `gradio.live` link. If your workspace blocks outbound
# MAGIC tunnels, that will fail — fall back to the driver-proxy URL printed below.

# COMMAND ----------

PORT = 8080
demo.launch(server_name="0.0.0.0", server_port=PORT, share=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Driver-proxy fallback URL
# MAGIC
# MAGIC If `share=True` didn't produce a link, build the in-workspace URL from the cluster context:

# COMMAND ----------

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
workspace_url = ctx.browserHostName().get()
org_id = ctx.workspaceId().get()
cluster_id = ctx.clusterId().get()
print(f"https://{workspace_url}/driver-proxy/o/{org_id}/{cluster_id}/{PORT}/")
