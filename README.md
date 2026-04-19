# rapidfireAI-rag-context-engineering

## Main Components
(Copied from [Project Statement](project1-statement.pdf))
- [ ] **Golden Data Creation:** Create at least 20 golden-labeled Q&A pairs over the documentation corpus, with identified source locations
- [ ] **System Building:** Build a RAG pipeline that, given a user question, retrieves relevant document chunks and generates an answer within the context budget
- [ ] **Experimentation and Evaluation:** Use RapidFire AI OSS to systematically compare multiple RAG configurations (varying chunking, embedding, retrieval, reranking, and prompting strategies) and analyze results

## Things to Do
- Golden Q&A Evaluation Data
  
  - Each pair must include:
    - A natural language question a real user might ask about the documentation
    - A solid reference answer
    - One or more source evidence locations, each as a filename and line number span
  
  - Question Types:
    - **Factual lookup** (e.g., “What arguments does RFGridSearch accept?”)
    - **Conceptual explanation** (e.g., “How does the adaptive execution engine differ from sequential execution?”)
    - **Procedural** (e.g., “How do I set up RapidFire AI for RAG evaluation?”)
    - **Comparative** (e.g., “What is the difference between RFGridSearch and RFRandomSearch?”)
  
  - Format:
    ```
    [
        {
            "question_id": xxx,
            "question": "...",
            "reference_answer": "...",
            "source_evidence": [
                {"file": "xxx.rst", "lines": [xx, xx]}
            ]
        }
    ]
    ```

- Automated RAG Pipeline
  
  To have a `main.py` that can be run as the following:

  ```
  python3 main.py --input input_filename --output output_filename
  ```

  - **Input File:** Test set `json` file containing a list of dictionary of questions
    - `question_id`: key with an integer value
    - `question`: key with a string value
    ```
    [
        {
            "question_id": /* id of the question */,
            "question": " /* question itself */"
        }
    ]
    ```
  
  - **Output File:** A `json` file containing examples as a list of dictionary
    - `question_id`: key with an integer value
    - `answer`: key with a string value
    - `soureces`: key with a list of tuples as value; each tuple has 3 entries: `(filename, startline, endline)`
    ```
    [
        {
            "question_id": /* corresponding question id */,
            "answer": "/* answer to the question */",
            "sources": [
                (/* file name */, /* start line */, /* end line */), 
                (/* file name */, /* start line */, /* end line */)
            ]
        }
    ]
    ```

- Report
  - **Data Creation Methodology**
  - **Final Pipeline Architecture**
  - **Experimentation Methodology**
  - **Results and Analysis**
  - **AI Tools Statement**

## Miscellaneous

> Notes from Discussion:
> 1. run `rapidfireai start` in the terminal before running any experiments to start the backend for a panel
> 2. to get the dataset for the discussion notebook, run `rapidfireai init --evals` in the terminal
> 3. [dashboard doc](https://oss-docs.rapidfire.ai/en/latest/icops.html)

---

```bash
RAPIDFIREAI-RAG-CONTEXT-ENGINEERING
├── README.md
├── project1-statement.pdf
├── golden_qa.json
├── main.py
├── data/
|   ├── validation-set
|   |   ├── validation-set-golden-qa-pairs.json
|   |   └── validation-set-question-type-distribution.json
|   └── sourcedocs/
├── misc/
|   ├── ai_usage.md
|   └── scratch.ipynb
└── discussion/
    ├── datasets/scifact/
    |   ├── corpus.jsonl
    |   ├── qrels.tsv
    |   ├── queries.jsonl
    |   └── test-queries.jsonl
    ├── discussion1-rag-scifact.ipynb
    └── rapidfireai_datahub_compat.py
```

---

