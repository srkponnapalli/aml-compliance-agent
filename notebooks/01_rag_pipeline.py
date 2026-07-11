# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — RAG Pipeline
# MAGIC
# MAGIC PDF ingestion → chunking → embeddings → in-memory vector store → retrieval → generation.
# MAGIC
# MAGIC Built from scratch (no LangChain, no vector database). Defines `documents`, `retrieve()`,
# MAGIC and `generate()`, which downstream notebooks pull in via `%run ./01_rag_pipeline`.

# COMMAND ----------

# MAGIC %pip install pypdf "typing_extensions>=4.12.0"

# COMMAND ----------

# Restart Python so the freshly installed libraries load cleanly. This wipes the session,
# so every import must come in the cells *after* this call.
#
# Guarded: only restart when this notebook is run standalone. When it's pulled in via `%run`
# from a parent (e.g. notebook 04 or the Gradio app), the parent has already installed +
# restarted, and restarting here would wipe the parent's session mid-run.
if not globals().get("RUNNING_AS_MODULE", False):
    dbutils.library.restartPython()

# COMMAND ----------

import re
from pypdf import PdfReader

# COMMAND ----------

import mlflow.deployments

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load the policy PDF
# MAGIC
# MAGIC Update `POLICY_PDF_PATH` to point at your workspace copy of the NorthStar policy manual.

# COMMAND ----------

POLICY_PDF_PATH = "/Workspace/Users/siva.ponnapalli@jrvs.ca/Learning_Srk/PDF/NorthStar_Compliance_Policy_Manual.pdf"

reader = PdfReader(POLICY_PDF_PATH)
full_text = ""
for page in reader.pages:
    full_text += page.extract_text()

print(f"Extracted {len(full_text)} characters from {len(reader.pages)} pages")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Chunking
# MAGIC
# MAGIC 500-character chunks with 100-character (20%) overlap, snapped to the nearest sentence
# MAGIC boundary so clauses aren't split mid-sentence.

# COMMAND ----------

def clean_text(text):
    return re.sub(
        r"\.{3,}|Page \| \d{1,3}|\n {3,}| \n{3,}|US IT \| Project Management Handbook",
        "",
        text,
    ).strip()


def split_text(text, chunk_size=500, overlap=100):
    text = clean_text(text)
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        pos = text.rfind(".", i, i + chunk_size)
        chunk = text[i:pos]
        chunks.append(chunk.strip())
    return chunks


chunks = split_text(full_text)
print(f"Produced {len(chunks)} chunks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Embeddings
# MAGIC
# MAGIC Databricks Foundation Model API (`databricks-bge-large-en`, 1024-dim). All chunks embedded
# MAGIC in a single call.

# COMMAND ----------

client = mlflow.deployments.get_deploy_client("databricks")

response = client.predict(
    endpoint="databricks-bge-large-en",
    inputs={"input": chunks},
)

embeddings = [item["embedding"] for item in response["data"]]
print(f"Embedded {len(embeddings)} chunks; dimension = {len(embeddings[0])}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. In-memory vector store
# MAGIC
# MAGIC Dict-of-dicts keyed by integer ID — O(1) lookup by ID at retrieval time.

# COMMAND ----------

documents = {}
for i in range(len(embeddings)):
    documents[i] = {"text": chunks[i], "embedding": embeddings[i]}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Cosine similarity (from scratch)

# COMMAND ----------

import math


def dot_product(a, b):
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    return sum(a[i] * b[i] for i in range(len(a)))


def magnitude(v):
    return math.sqrt(dot_product(v, v))


def cosine_similarity(a, b):
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    return dot_product(a, b) / (magnitude(a) * magnitude(b))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Retrieval

# COMMAND ----------

def retrieve(query, documents, top_k=3):
    query_embedding = client.predict(
        endpoint="databricks-bge-large-en",
        inputs={"input": query},
    )["data"][0]["embedding"]

    scores = {}
    for i in documents:
        scores[i] = {
            "similarity": cosine_similarity(documents[i]["embedding"], query_embedding),
            "text": documents[i]["text"],
        }

    sorted_scores = sorted(scores.items(), key=lambda x: x[1]["similarity"], reverse=True)
    return [score[1]["text"] for score in sorted_scores[:top_k]]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Generation

# COMMAND ----------

def generate(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
    messages = [
        {"role": "system", "content": "You are a compliance assistant for NorthStar Financial Services."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"},
    ]
    response = client.predict(
        endpoint="databricks-meta-llama-3-3-70b-instruct",
        inputs={"messages": messages},
    )
    return response

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Smoke test
# MAGIC
# MAGIC Skipped when this notebook is pulled in via `%run` (i.e. `RUNNING_AS_MODULE` is set).

# COMMAND ----------

if not globals().get("RUNNING_AS_MODULE", False):
    query = "What is the threshold for filing a Currency Transaction Report?"
    chunks_retrieved = retrieve(query, documents, top_k=3)
    answer = generate(query, chunks_retrieved)
    print(answer["choices"][0]["message"]["content"])
