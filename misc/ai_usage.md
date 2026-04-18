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

2. 