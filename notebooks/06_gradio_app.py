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
    "violation_sweep": "🔍 Violation Sweep (fixed AML checks + policy)",
    "text_to_sql": "🧮 Text-to-SQL (ad-hoc query)",
    "policy_lookup": "📖 Policy Lookup (RAG over the policy manual)",
}

EXAMPLES = [
    "Are there any compliance violations in our recent transactions?",
    "How many wire transfers were sent to each destination country?",
    "What is the threshold for filing a Currency Transaction Report?",
]


def respond(question):
    if not question or not question.strip():
        return "", "Please enter a question."
    tool, answer = agent(question)
    return TOOL_LABELS.get(tool, tool), answer


with gr.Blocks(title="NorthStar Compliance Agent") as demo:
    gr.Markdown(
        "# NorthStar Compliance Agent\n"
        "Ask about transactions or policy. The agent **routes** your question to the right tool "
        "and shows you which one it picked."
    )
    question = gr.Textbox(label="Question", placeholder="e.g. Are there any compliance violations?", lines=2)
    ask = gr.Button("Ask", variant="primary")
    tool_used = gr.Textbox(label="Tool selected by the router", interactive=False)
    answer = gr.Markdown(label="Answer")

    ask.click(respond, inputs=question, outputs=[tool_used, answer])
    question.submit(respond, inputs=question, outputs=[tool_used, answer])
    gr.Examples(EXAMPLES, inputs=question)

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
