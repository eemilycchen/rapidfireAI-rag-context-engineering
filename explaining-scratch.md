# Explaining `scratch.ipynb`

This notebook builds and evaluates a RAG (Retrieval-Augmented Generation) pipeline over
RapidFire AI's `.rst` documentation files using the RapidFire AI framework and LangChain.

## Setup (Cells 0–4)

- Loads the Triton API key from `~/api-key.txt` and sets the base URL for the UCSD Triton AI API endpoint.
- Imports standard libraries (`re`, `json`, `defaultdict`, `pathlib`, HuggingFace `datasets`).
- Imports LangChain components for document loading, text splitting, embeddings, retrieval, and cross-encoder reranking.
- Imports RapidFire AI classes: `List`, `RFLangChainRagSpec`, `RFOpenAIAPIModelConfig`, `RFPromptManager`, `RFGridSearch`, and `Experiment`.

## Document Loading (Cells 5–9)

- Defines `TextLoaderWithMetadata`, a custom subclass of LangChain's `TextLoader` that accepts a `metadata_func` callback. The callback is applied to each loaded document's metadata during both eager and lazy loading.
- Defines `metadata_func`, which strips directory paths from the `source` field so metadata contains only the filename (e.g. `models.rst`).
- Creates a `DirectoryLoader` that reads all `*.rst` files from `../data/sourcedocs/` using the custom loader with multithreading enabled.
- Loads 31 documents and verifies the metadata format.

## Text Splitting (Cells 10–12)

- Defines `TextSplitterWithLineNum`, a custom subclass of `RecursiveCharacterTextSplitter`. After splitting, it maps each chunk back to its original line numbers in the source file by building a reverse index of stripped line content to line numbers. Each chunk's `source` metadata is rewritten from a plain filename to `{"file": <filename>, "lines": [start, end]}`.
- Instantiates the splitter using the `o200k_base` tiktoken encoding with a chunk size of 512 tokens and 64-token overlap.

## RAG Pipeline Configuration (Cells 13–20)

- **Embedding**: Uses `sentence-transformers/all-MiniLM-L6-v2` via `HuggingFaceEmbeddings` on CPU with normalized embeddings and a batch size of 32. An OpenAI embedding option is commented out.
- **Vector store**: FAISS (in-memory, CPU-only) with batch size 32.
- **Search**: Cosine similarity search retrieving `k=10` documents. An MMR alternative is commented out.
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L6-v2` via `CrossEncoderReranker`, keeping the top 5 results after reranking.
- **Document template**: A custom function that formats each chunk as `{source}: {page_content}` for injection into prompts.
- All of these are assembled into an `RFLangChainRagSpec` instance.

## System Prompt (Cell 21)

Defines `INSTRUCTIONS`, the system-level prompt for the generator LLM. It instructs the model to:

1. Answer using only the retrieved documentation context.
2. Keep answers concise, in plain text (no markdown).
3. Cite relevant filenames with start/end line numbers.
4. Output in the format: `#### ANSWER: <answer>` followed by `#### SOURCE: [{"file": ..., "lines": [...]}]`.

## Evaluation Dataset (Cells 22–24)

- `load_eval_question_set` loads a JSON file of questions into a HuggingFace `Dataset`. For validation sets, it also loads `reference_answer` and `source_evidence` fields.
- Loads the golden QA validation set (45 questions) from `../data/validation-set/validation-set-golden-qa-pairs.json`.

## Preprocessing Function (Cell 25)

`sample_preprocess_fn` is called per batch during evaluation:

1. Retrieves context documents for each question via `rag.get_context()`.
2. Extracts source metadata and page content from each retrieved document.
3. Serializes retrieved documents into plain text strings.
4. Builds chat-style prompts (system message with instructions + user message with question and context).
5. Returns the batch augmented with `prompts`, `retrieved_documents`, and `retrieved_contexts`.

## Postprocessing and Answer Extraction (Cells 26–27)

- `extract_solution` uses regex to parse `#### ANSWER: ...` and `#### SOURCE: ...` markers from the model's raw generated text, returning the answer and source as separate values.
- `sample_postprocess_fn` applies `extract_solution` to each generated text in the batch, adding `answer` and `source` fields.

## Model and Experiment Configuration (Cells 28–31)

- Configures `RFOpenAIAPIModelConfig` pointing at the Triton API (`api-mistral-small-3.2-2506`) with rate limits of 120 RPM and 1M TPM.
- Defines placeholder `compute_metrics_fn` and `accumulate_metrics_fn` that return empty dictionaries (metrics to be added later).
- Wraps everything in `RFGridSearch` with a single configuration (one embedding model, one search type, one generator).
- Creates an `Experiment` in `evals` mode, runs `run_evals()` with 4 shards and seed 42 over the golden QA validation set, then ends the experiment. The experiment processes all 45 questions across 4 shards, generating answers with the Mistral model.

## Results Inspection (Cells 32–33)

- Inspects the structure of `results`: a dict keyed by run/config ID, containing a tuple of aggregated and cumulative metrics dictionaries. Each entry has keys like `question_id`, `question`, `reference_answer`, `source_evidence`, `answer`, `source`, `retrieved_documents`, and `retrieved_contexts`.
- Iterates over all questions to print side-by-side comparisons of:
  - The ground-truth reference answer and source evidence.
  - The model's generated answer and cited sources.
  - The retrieved context text corresponding to each cited source (matched by looking up the source metadata in the retrieved documents list).

**Known issue**: The source-to-context matching uses `list.index()` for exact dictionary equality, which fails when the model's cited line ranges don't exactly match the retrieved document metadata (the `ValueError` at question 26). A fuzzy match on filename would resolve this.
