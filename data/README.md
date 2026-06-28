# data/

```
NorthStar_Compliance_Policy_Manual.pdf
```

`notebooks/01_rag_pipeline.py` reads this PDF (see the `POLICY_PDF_PATH` variable) to build the
RAG index. The document is fully synthetic and committed to the repo. When running on Databricks,
point `POLICY_PDF_PATH` at the file's location in your workspace / Repos checkout.
