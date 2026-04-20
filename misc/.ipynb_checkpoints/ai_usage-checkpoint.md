# AI used in the Project

1. Model: ChatGPT
   
   ---
   Prompt:  
   > what are the `encoding_name` options for `RecursiveCharacterTextSplitter.from_tiktoken_encoder`  and what are the differences
   ---

   Notes from the response:

   Models:
   - `gpt2` / `r50k_base`: older ~50k-vocab GPT-2 style tokenizers. `r50k_base` is the standard GPT-2 BPE; `gpt2` is the GPT-2 encoding entry in `tiktoken`. Both use the same regex pattern and the same end-of-text token, with `gpt2` built from the original GPT-2 encoder/vocab files
   - `p50k_base`: also ~50k vocab, but different merge ranks from `r50k_base`; intended for Codex-style/code-heavy tokenization
   - `p50k_edit`: same base merges as `p50k_base`, but adds extra FIM/edit special tokens like `<|fim_prefix|>`, `<|fim_middle|>`, and `<|fim_suffix|>`
   - `cl100k_base`: newer ~100k-vocab tokenizer with a different regex and a larger special-token set; commonly associated with GPT-4 / GPT-3.5-era model families
   - `o200k_base`: newer ~200k-vocab tokenizer with a different regex again; used by newer model families such as `gpt-4o`
   - `o200k_harmony`: derived from `o200k_base` but with many additional special tokens layered on top
   
   Practical Differences:
   - They split text differently
   - Token counts for the same text can change
   - Special-token behavior differs
   - Best choice depends on the model you actually use

2. Model: ChatGPT
   
   ---
   Prompt: 
   > when using RFLangChainRagSpec it can take a retriever 
   > 1. what is that retriever used for 
   > 2. what are the options that i can use for it 
   > 3. what are the differences
   ---

   Notes from the response: 

   Retriever: [Source](https://oss-docs.rapidfire.ai/en/latest/ragspecs.html)

   `retriever` in `RFLangChainRagSpec` is the component that actually fetches the context documents/chunks for each query during the query-processing stage. In RapidFire’s control flow, if you pass `retriever`, RapidFire uses it as-is and does not build one from `document_loader`, `text_splitter`, or `vector_store_cfg`. If you do not pass a retriever, RapidFire will either create one from `vector_store_cfg.as_retriever()`, or fall back to building a default FAISS-based retriever from your documents.

   It is the retrieval engine for RAG: given the eval/query text, it returns the relevant documents or chunks that become the model’s context. In LangChain terms, a retriever is “an interface that returns documents given an unstructured query.” In RapidFire, it sits before optional reranking, and RapidFire’s docs explicitly say that when `retriever` is supplied, that object is the one used for chunk retrieval.

   Options: [Source](https://reference.langchain.com/python/langchain-classic/retrievers/self_query/base/SelfQueryRetriever?utm_source=chatgpt.com)

   At the RapidFire API level, the type is a LangChain `BaseRetriever`, so you can pass any LangChain retriever implementation that conforms to that interface. RapidFire’s docs do not restrict it to a fixed whitelist.

   - Pass no retriever and let RapidFire create one:
     - From `vector_store_cfg` via LangChain `as_retriever()` RapidFire supports FAISS, PGVector, and Pinecone vector-store configs for that path.
     - With search_cfg.type set to:
       - `similarity`
       - `similarity_score_threshold`
       - `mmr` plus knobs like `k`, `score_threshold`, `fetch_k`, and `lambda_mult`
   - Pass a custom LangChain retriever:
     - Plain vector-store retriever from `vectorstore.as_retriever(...)`:
       - Best for: standard semantic search
       - Pros: simplest, fastest, least moving parts
       - Cons: weakest when queries need reformulation, metadata reasoning, or post-filtering In RapidFire, this is what you effectively get from `vector_store_cfg + search_cfg`
     - MMR / threshold search via `search_cfg`:
       - `similarity`: highest semantic similarity only
       - `similarity_score_threshold`: only keep docs above a score cutoff
       - `mmr`: trades some raw similarity for diversity in results Use this when you do not need a fully custom retriever class, just different search behavior on a vector store.
     - `SelfQueryRetriever`:
       - Difference: uses an LLM to turn the user query into a structured search, often including metadata filters
       - Good when your corpus has useful metadata and users ask things like “recent security docs about auth” rather than pure semantic lookup.
     - `ContextualCompressionRetriever`:
       - Difference: retrieve first, then trim/filter the returned docs based on the query
       - Good when retrieved docs are long/noisy and token budget matters.
     - `MultiQueryRetriever`
       - Difference: expands one query into several paraphrases/perspectives, then merges unique hits
       - Good when user questions are underspecified or embeddings miss relevant phrasing.
       - Tradeoff: more retrieval work and often more latency/cost because it uses an LLM.
     - `EnsembleRetriever`:
       - Difference: combines several retrievers, such as dense vector + lexical/BM25-style retrieval, with rank fusion
       - Good when one retrieval method alone misses results.
       - Tradeoff: more complexity and latency.
     - `MultiVectorRetriever`:
       - Difference: one logical document can be represented by multiple vectors
       - Good when you want richer recall via chunk vectors, summaries, generated questions, etc.
  
   Practical Recommendation:
   - Use no explicit retriever when a normal vector-store retriever is enough; just tune `vector_store_cfg` and `search_cfg`
   - Pass a custom `retriever` when you specifically want behavior like **multi-query** **expansion**, **compression**, **metadata-aware self-querying**, **hybrid/ensemble retrieval**, or **parent-doc retrieval**.
  
3. 