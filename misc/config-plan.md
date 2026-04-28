## What we'll build

Two pieces, in one notebook (or split into a notebook + helper module):

1. A grid-search runner that defines 8 RAG configs via `RFGridSearch` + `RFLangChainRagSpec` over the RapidFire docs corpus
2. A results-tracker that reads RapidFire's outputs after `run_evals` finishes and writes a clean markdown summary file

## Files to create / modify

### Existing files we'll leverage
- [data/sourcedocs/](data/sourcedocs/) - the documentation corpus (~50 `.rst` files) loaded via LangChain `DirectoryLoader`
- [data/validation-set/validation-set-golden-qa-pairs.json](data/validation-set/validation-set-golden-qa-pairs.json) - eval set
- [discussion/discussion1-rag-scifact.ipynb](discussion/discussion1-rag-scifact.ipynb) - reference patterns for `RFLangChainRagSpec`, `preprocess_fn`, etc.

### New files
- `experiments/grid_search.ipynb` (or `.py`) - main grid-search runner
- `experiments/track_results.py` - helper module to dump results
- `logs/experiment_results.md` - **auto-generated** human-readable results table (the file you asked about)
- `logs/experiment_results.csv` - same data in CSV (handy for plotting/Excel)
- `logs/experiment_results.json` - raw structured dump (for re-analysis)

## Grid-search design (8 configs across 4 axes)

Four axes varied: **chunk size, embedding model, retriever type, LLM model**. Reranker and search settings held constant (`CrossEncoderReranker` ms-marco-MiniLM-L6, `top_n=5`, `similarity` search, `k=10`) so we isolate the four dimensions of interest.

### Embedding models — source of truth (per project statement)

The course points you to the official model list: [TritonAI model hub table](https://tritonai-api.ucsd.edu/ui/model_hub_table/). That page may list **chat** models prominently; **embedding** models might appear as separate rows, under a filter, or only in the raw API.

**Resolution before implementation:**

1. On the hub, look for any model whose name or type suggests **embeddings** (e.g. contains `embed`, or a “mode”/category column = embedding). If you find one, use `OpenAIEmbeddings` with `base_url`/`api_key` pointing at Triton and that model id for runs **5** and **8** (Triton-embedding ablation vs HF mpnet).
2. If **no** Triton embedding model appears, **do not** guess `api-tgpt-embeddings`. Keep the 8 runs meaningful by swapping the “second embedding” to a **different HuggingFace** model (e.g. `BAAI/bge-small-en-v1.5` or `intfloat/e5-base-v2` vs `sentence-transformers/all-mpnet-base-v2`). The report still says you compared “meaningfully distinct” embedding families (smaller fast vs larger better, or E5 vs MPNet, etc.).

Optional double-check: the OpenAI-style `GET /v1/models` on the same gateway (with `Authorization: Bearer <key>`) can list ids not obvious on the UI.

### LLM models (Triton; confirmed in planning)

- **Weak:** `api-mistral-small-3.2-2506` (already used in existing code; good baseline).
- **Strong:** `api-llama-4-scout` or `api-gpt-oss-120b` (pick one; both appear on the hub snapshot shared in chat).

| Run | Chunk (size/overlap) | Embedding | Retriever | LLM Model | Purpose / Ablation |
|-----|----------------------|-----------|-----------|-----------|---------------------|
| 1 | 256 / 32 | HF mpnet-base-v2 | dense (FAISS) | `api-mistral-small-3.2-2506` | baseline |
| 2 | 512 / 64 | HF mpnet-base-v2 | dense (FAISS) | `api-mistral-small-3.2-2506` | chunk size effect (vs 1) |
| 3 | 256 / 32 | HF mpnet-base-v2 | dense (FAISS) | strong (e.g. `api-llama-4-scout`) | LLM strength on small chunks (vs 1) |
| 4 | 512 / 64 | HF mpnet-base-v2 | dense (FAISS) | strong | LLM on big chunks (vs 2); dense baseline for hybrid comparison |
| 5 | 512 / 64 | Triton embedding **if listed on hub**; else HF alt (e.g. bge-small) | dense (FAISS) | strong | embedding upgrade (vs 4) |
| 6 | 512 / 64 | HF mpnet-base-v2 | hybrid (BM25 + dense, weights 0.5/0.5) | strong | hybrid lift over pure dense (vs 4) |
| 7 | 256 / 32 | HF mpnet-base-v2 | hybrid (BM25 + dense) | strong | hybrid + small chunks (vs 6, vs 3) |
| 8 | 512 / 64 | same embedding as run 5 | hybrid (BM25 + dense) | strong | best-of-everything (vs 5, vs 6) |

### Pairwise ablations this enables (great for the report's "what knob mattered most" analysis)

- **1 vs 2**: chunk 256 vs 512 (weak LLM, dense, mpnet)
- **1 vs 3**: weak vs strong LLM (chunk 256, dense, mpnet)
- **2 vs 4**: weak vs strong LLM (chunk 512, dense, mpnet)
- **4 vs 5**: second embedding family (mpnet vs Triton embed if available, else mpnet vs e.g. bge-small) — chunk 512, dense, strong LLM
- **4 vs 6**: dense vs hybrid (chunk 512, mpnet, strong LLM) — the hybrid retrieval headline
- **6 vs 7**: chunk 512 vs 256 (hybrid, mpnet, strong LLM)
- **5 vs 8**: dense vs hybrid (same “second” embedding as run 5, chunk 512, strong LLM)
- **6 vs 8**: mpnet vs same embedding as run 5 (hybrid, chunk 512, strong LLM)

Each comparison answers a counterfactual the TAs may ask in the F2F (`project1-statement-updated.pdf` p.6: *"What would happen if you changed X?"*).

### Why this design beats the original

Per the project rubric (`project1-statement-updated.pdf` p.5): *"Which knobs had the biggest impact on retrieval metrics vs. generation metrics?"*

- **Embedding + retriever** axes drive **retrieval score** (F1@5, P@5, R@5)
- **LLM model** axis drives **generation score** (Correctness, Faithfulness, Completeness)
- **Chunk size** affects both (smaller = more precise retrieval but more context fragmentation; larger = better context coherence but lower retrieval precision)

So the 8 configs span all 4 sub-scores of the leaderboard formula in a controlled way.

### Implementation notes

- **Configs 1-5** use the built-in `vector_store_cfg={"type": "faiss"}` path — set `text_splitter`, `embedding_cfg`, `vector_store_cfg`, `search_cfg`, `reranker_cfg` directly on `RFLangChainRagSpec`.
- **Configs 6-8 (hybrid)** require the custom `retriever=` path (`data/sourcedocs/ragspecs.rst:100-101`, 368). For each, we pre-build chunks → BM25Retriever + FAISS dense retriever → wrap in LangChain `EnsembleRetriever(weights=[0.5, 0.5])` → pass via `retriever=`. When `retriever` is set, `document_loader`/`text_splitter`/`vector_store_cfg` are not required (they're handled by us in the helper).
- **LLM model** is varied via `RFOpenAIAPIModelConfig(model_config={"model": "..."})` per config (`data/sourcedocs/configs.rst:129-160`). Weak = `api-mistral-small-3.2-2506`, strong = `api-llama-4-scout` or `api-gpt-oss-120b` per [model hub](https://tritonai-api.ucsd.edu/ui/model_hub_table/).
- **Embeddings:** If the hub lists a Triton embedding model id, use `OpenAIEmbeddings` with that id for runs 5 and 8. Otherwise use two HF `embedding_cfg` variants only (no guesswork).
- Configs are assembled as a `List` of full eval-config dicts wrapped in `RFGridSearch(config_set)` (`data/sourcedocs/configs.rst:66-93`), since configs differ on too many axes for nested `List(...)` per knob.

### Cost / budget strategy

Strong-LLM configs (3-8) consume more tokens. Mitigations:
- Run grid on **downsampled validation set** (e.g., 30 questions out of the released set) for the experimentation phase
- Cache FAISS indices per `(chunk_size, embedding)` pair so we don't re-embed across runs that share embedding+chunk
- Use RapidFire's IC Ops (`Stop`/`Clone-Modify`) to kill clearly-poor runs early (`data/sourcedocs/onlineagg.rst`)

## Results-tracker logic (the txt/markdown file you asked about)

After `experiment.run_evals(...)` completes, `track_results.py` does the following:

```python
def dump_results(experiment, output_dir="logs", config_labels=None):
    runs_info = experiment.get_runs_info()       # config per run
    results   = experiment.get_results()         # metrics per run/step
    
    # Pick the FINAL row per run (last step)
    final = results.sort_values("step").groupby("run_id").tail(1)
    merged = runs_info.merge(final, on="run_id")
    
    # Write 3 outputs: markdown, CSV, JSON
    merged.to_csv(f"{output_dir}/experiment_results.csv", index=False)
    merged.to_json(f"{output_dir}/experiment_results.json", orient="records", indent=2)
    
    # Build a pretty markdown table from `merged`
    write_markdown_table(merged, f"{output_dir}/experiment_results.md")
```

The markdown file will look like:

```markdown
# RapidFire AI RAG Experiment Results
Generated: 2026-04-27 14:30 PT

## Config Knob Table

| Run | Chunk | Embedding | Retriever | LLM | Reranker top_n | Notes |
|-----|-------|-----------|-----------|-----|----------------|-------|
| 1 | 256/32 | mpnet-base | dense | mistral-small | 5 | baseline |
| 2 | 512/64 | mpnet-base | dense | mistral-small | 5 | chunk effect |
| 3 | 256/32 | mpnet-base | dense | llama-4-scout | 5 | LLM upgrade |
| 4 | 512/64 | mpnet-base | dense | llama-4-scout | 5 | dense + strong baseline |
| 5 | 512/64 | Triton embed or HF bge-small | dense | llama-4-scout | 5 | embed ablation |
| 6 | 512/64 | mpnet-base | hybrid (BM25+dense) | llama-4-scout | 5 | hybrid lift |
| 7 | 256/32 | mpnet-base | hybrid | llama-4-scout | 5 | hybrid + small chunks |
| 8 | 512/64 | same as 5 | hybrid | llama-4-scout | 5 | best-of-everything |

## Metrics per Run

| Run | F1@5 | Precision@5 | Recall@5 | Correctness | Faithfulness | Completeness | Leaderboard |
|-----|------|-------------|----------|-------------|--------------|--------------|-------------|
| 1 | 0.42 | 0.55 | 0.34 | 0.80 | 0.72 | 3.4 | 0.58 |
| ... | ... | ... | ... | ... | ... | ... | ... |

## Best Config
Run 8 — Leaderboard 0.78 (chunk=512, second embedding family, hybrid retriever, strong Triton LLM)
```

This satisfies the report's "table of all config knob details" requirement directly — you can copy it straight into your project report.

## How it ties to the project's `logs/` requirement

`project1-statement-updated.pdf` page 6 explicitly says:
> "subfolder named logs with all your RapidFire AI experiment logs and metrics"
> "RapidFire AI outputs them as rapidfire.log; all metrics files are written to your experiments_path directory"

So we'll set `experiments_path="logs/"` in the `Experiment(...)` constructor — that puts `rapidfire.log` and all metrics directly in the right folder, and our markdown summary lives alongside them.

## Optional: README.md update

The tracker can optionally append/replace a "Latest Experiment Results" section in `README.md` so anyone browsing the repo on GitHub immediately sees the table. Disabled by default; enable with `dump_results(..., update_readme=True)`.

## Open questions — confirm before I implement

1. **Triton embedding model**: On [model hub](https://tritonai-api.ucsd.edu/ui/model_hub_table/), note whether any row is clearly an **embedding** model. If yes, record its exact `id` for runs 5 and 8. If no, we use two **HuggingFace** embedding models for the embedding ablation (plan already allows this).
2. **Strong LLM**: Default `api-llama-4-scout`; use `api-gpt-oss-120b` instead if you prefer (both were on your hub snapshot).
3. **Hybrid retriever weights**: Default `[0.5, 0.5]` for BM25 + dense.
4. **Downsample size**: Default 30 validation questions for experimentation; full set for final run.

Once you confirm (or just say "go"), I switch to agent mode and build:
- `experiments/grid_search.ipynb` (the runner)
- `experiments/track_results.py` (the tracker that creates `logs/experiment_results.md`)
- `experiments/build_hybrid_retriever.py` (helper for configs 6-8)
- `logs/` outputs auto-populated when the grid runs

## Recommendations from Project Statement and Discussion Slides

| Axis                  | Recommended by project? | You already vary it? | What to add                                                                  |
| --------------------- | ----------------------: | -------------------: | ---------------------------------------------------------------------------- |
| Chunking              |                     Yes |                  Yes | Add `384/64` as a middle value                                               |
| Embedding model       |                     Yes |                  Yes | Keep HF mpnet vs BGE/E5 or Triton embedding if available                     |
| Retrieval type        |                     Yes |                  Yes | Dense vs BM25 vs hybrid                                                      |
| Search algorithm      |                     Yes |           Not really | Try `similarity`, `MMR`, maybe `score_threshold`                             |
| Top-k                 |                     Yes |        Held constant | Try `k=5`, `k=10`, `k=15`                                                    |
| Reranking             |                     Yes |        Held constant | Try no reranker vs reranker; `top_n=3` vs `top_n=5`                          |
| Prompting             |                     Yes |              Not yet | Try strict grounded prompt vs concise prompt vs source-aware prompt          |
| Vector store backend  |  Yes, mostly efficiency |              Not yet | FAISS is fine; PGVector/Pinecone mainly help if you want persistent indexing |
| Context serialization |                 Implied |              Not yet | Try metadata-rich chunks vs plain chunks                                     |
