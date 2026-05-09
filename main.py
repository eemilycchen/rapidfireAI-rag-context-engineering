import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict
from typing import List as listtype

from datasets import Dataset

# import custom modules
from modules import TextLoaderWithMetadata, TextSplitterWithLineNum

# document loaders
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# embeddings
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

# cross encoder for reranking
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

# rapidfire imports
from rapidfireai.automl import (
    List,
    RFLangChainRagSpec,
    RFOpenAIAPIModelConfig,
    RFPromptManager,
    RFGridSearch,
)
from rapidfireai import Experiment

DEFAULT_BASE_URL = 'https://tritonai-api.ucsd.edu'
batch_size = 32

INSTRUCTIONS = """
You are a helpful assistant answering questions about RapidFire AI docuentation.

Your job:
- Read the retrieved documentation context
- Answer the user's question using only the context
- Do not make up information that is not supported by the retrieved text

Rules:
1. Use only the retrieved document context.
2. Do not infer, generalize, or add plausible details that are not explicitly present.
3. The answer should be in plain text and do not use any markdown syntax including list.
4. Mention relevant file names and the corresponding start and end line number when they are clearly useful.
5. There can be multiple sources but only keep the ones that are really relevant.
6. The sources must be in valid JSON format.
7. Do not add background knowledge or plausible details that are not in the context.
8. Answer all parts of the question, including parameters, defaults, constraints, counts, examples, and workflow steps when they appear in the context.
9. Every factual claim must be directly supported by the retrieved context.

Output format:
#### ANSWER: <final answer>
#### SOURCE: [{"file": <source file>, "lines": [<start line>, <end line>]}]
"""


def _read_api_key(path: Path) -> str:
    text = path.expanduser().read_text(encoding='utf-8')
    line = text.splitlines()[0].strip() if text.strip() else ''
    if not line:
        raise ValueError(f'API key file is empty: {path}')
    return line


def _is_rate_limit_error(err: BaseException) -> bool:
    """Best-effort detection of HTTP 429 / rate limiting for fallback generation."""
    if err is None:
        return False
    cur: BaseException | None = err
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        msg = f'{type(cur).__name__} {cur}'.lower()
        if '429' in msg or 'rate limit' in msg or 'too many requests' in msg:
            return True
        cur = getattr(cur, '__cause__', None) or getattr(cur, '__context__', None)
    return False


def _model_config_for_gateway(model_name: str) -> dict:
    cfg: dict = {'model': model_name, 'max_completion_tokens': 2048}
    lower = model_name.lower()
    if 'gpt-oss' in lower or 'oss-120' in lower:
        cfg['reasoning_effort'] = 'high'
    return cfg


def _build_config_group(*, corpus_dir: Path, api_key: str, base_url: str, generation_model: str):
    loader = DirectoryLoader(
        path=str(corpus_dir),
        glob='**/*.rst',
        loader_cls=TextLoaderWithMetadata,
        loader_kwargs={'metadata_func': metadata_func},
        show_progress=True,
        use_multithreading=True,
    )
    text_splitter = TextSplitterWithLineNum().from_tiktoken_encoder(
        encoding_name='o200k_base', chunk_size=384, chunk_overlap=64, separators=['\n', '\n\n']
    )
    embedding_cfg = {
        'class': OpenAIEmbeddings,
        'model': 'api-tgpt-embeddings',
        'api_key': api_key,
        'base_url': base_url,
        'check_embedding_ctx_length': False,
    }
    vector_store_cfg = {'type': 'faiss', 'batch_size': batch_size}
    search_cfg = List([{'type': 'mmr', 'k': 15}])
    reranker_cfg = {
        'class': CrossEncoderReranker,
        'model_name': 'BAAI/bge-reranker-v2-m3',
        'model_kwargs': {'device': 'cpu'},
        'top_n': 8,
    }
    rag = RFLangChainRagSpec(
        document_loader=loader,
        text_splitter=text_splitter,
        embedding_cfg=embedding_cfg,
        vector_store_cfg=vector_store_cfg,
        search_cfg=search_cfg,
        reranker_cfg=reranker_cfg,
        enable_gpu_search=False,
        document_template=custom_template,
    )
    openai_config = RFOpenAIAPIModelConfig(
        client_config={'api_key': api_key, 'base_url': base_url, 'max_retries': 2},
        model_config=_model_config_for_gateway(generation_model),
        rpm_limit=120,
        tpm_limit=1_000_000,
        rag=rag,
        prompt_manager=None,
    )
    config_set = List([{
        'openai_config': List([openai_config]),
        'batch_size': batch_size,
        'preprocess_fn': sample_preprocess_fn,
        'postprocess_fn': sample_postprocess_fn,
        'compute_metrics_fn': {},
        'accumulate_metrics_fn': {},
    }])
    return RFGridSearch(config_set)


def main():
    parser = argparse.ArgumentParser(description='RAG eval generation via RapidFire AI.')
    parser.add_argument('--input', required=True, help='JSON file of [{question_id, question}, ...]')
    parser.add_argument('--output', required=True, help='Where to write predictions JSON')
    parser.add_argument(
        '--corpus-dir',
        required=True,
        help='Root of the .rst corpus (released sourcedocs/ layout)',
    )
    parser.add_argument(
        '--apikey-txt',
        required=True,
        help='Path to text file containing the TritonAI gateway API key (first line)',
    )
    parser.add_argument(
        '--generation-model',
        required=True,
        help='Gateway model id for answer generation (e.g. api-gpt-oss-120b, claude-sonnet-4-6)',
    )
    parser.add_argument(
        '--fallback-generation-model',
        default='api-mistral-small-3.2-2506',
        help='If the primary model hits rate limits (429), retry the run with this model.',
    )
    parser.add_argument(
        '--base-url',
        default=DEFAULT_BASE_URL,
        help=f'OpenAI-compatible gateway base URL (default: {DEFAULT_BASE_URL})',
    )
    args = parser.parse_args()

    api_key = _read_api_key(Path(args.apikey_txt))
    corpus_dir = Path(args.corpus_dir).expanduser().resolve()
    base_url = str(args.base_url).rstrip('/')
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f'--corpus-dir is not a directory: {corpus_dir}')

    models_to_try = []
    for m in (args.generation_model, args.fallback_generation_model):
        if m and m not in models_to_try:
            models_to_try.append(m)

    questions = load_question_set(args.input)

    for i, model_name in enumerate(models_to_try):
        experiment = None
        try:
            config_group = _build_config_group(
                corpus_dir=corpus_dir,
                api_key=api_key,
                base_url=base_url,
                generation_model=model_name,
            )
            experiment = Experiment(experiment_name='rag-generation-only', mode='evals')
            results = experiment.run_evals(
                config_group=config_group, dataset=questions, num_shards=4, seed=42
            )
            print('experiments finished')
            write_answer(args.output, results)
            break
        except Exception as e:
            if i < len(models_to_try) - 1 and _is_rate_limit_error(e):
                print(
                    f'Rate limit error for {model_name}; retrying with {models_to_try[i + 1]} …',
                    file=sys.stderr,
                )
                continue
            raise
        finally:
            if experiment is not None:
                try:
                    experiment.end()
                except Exception:
                    pass


def metadata_func(document, metadata):
    metadata['source'] = Path(metadata['source']).name
    return metadata


def custom_template(doc):
    return f'{doc.metadata["source"]}: {doc.page_content}'


def load_question_set(path, encoding='utf-8'):
    # load the questions from the file
    with open(path, 'r', encoding=encoding) as f:
        eval_questions = json.load(f)
    # to store each questions
    questions = []
    for example in eval_questions:
        # extract the question and id and store into the list
        row = {
            'question_id': example['question_id'],
            'question': example['question']
        }
        questions.append(row)
    # convert into a hugging face dataset
    eval_dataset = Dataset.from_list(questions)
    return eval_dataset


def write_answer(path, results):
    print('writing answers to the output file')
    # question id
    question_id = results[1][0]['question_id']
    # model generated output
    answers, sources = results[1][0]['answer'], results[1][0]['source']
    # retrieved context
    retrieved_documents = results[1][0]['retrieved_documents']
    retrieved_contexts = results[1][0]['retrieved_contexts']

    output_answers = []

    # for each question
    for i in range(len(question_id)):
        # extract the answers including id, answer, source, and retrieved context
        answer = {}
        answer['question_id'] = question_id[i]
        answer['answer'] = answers[i].strip()
        if sources[i] is None:
            # fallback to retrieved documents if the model generated source fails to parse
            answer['sources'] = retrieved_documents[i][:2]
        else:
            try:
                model_source = json.loads(sources[i])
                answer['sources'] = []
                # check if the model generated sources are in the retrieved documents
                for s in model_source:
                    if s in retrieved_documents[i]:
                        answer['sources'].append(s)
                # fallback to retrieved documents if the model generated sources are not in the retrieved documents
                if len(answer['sources']) == 0:
                    answer['sources'] = retrieved_documents[i][:2]
            except json.JSONDecodeError:
                # fallback to retrieved documents if the model generated sources can not be decoded
                print(f'{sources[i]} can not decode the model generated source')
                answer['sources'] = retrieved_documents[i][:2]
        answer['retrieved_context'] = '\n\n'.join(retrieved_contexts[i][:5])
        # add to the final answer list
        output_answers.append(answer)
    # write the answers to the output file
    with open(path, 'w') as f:
        json.dump(output_answers, f, indent=4)


def sample_preprocess_fn(batch: Dict[str, listtype], 
                         rag: RFLangChainRagSpec, 
                         prompt_manager: RFPromptManager) -> Dict[str, listtype]:
    '''
    convert a batch of dataset rows into model-ready prompts
    '''
    # retrieve context for each question in the batch
    all_context = rag.get_context(batch_queries=batch['question'], serialize=False)
    # extract the source metadata field and retrieved context from each retrieved document
    retrieved_documents, retrieved_contexts = [], []
    for docs in all_context:
        evidence_for_query, context = [], []
        for doc in docs:
            evidence_for_query.append(doc.metadata.get('source'))
            context.append(doc.page_content)
        # attach the source evidence and retrieved context to the question
        retrieved_documents.append(evidence_for_query)
        retrieved_contexts.append(context)

    # convert retrieved docs into plain text strings suitable for prompt injection
    serialized_context = rag.serialize_documents(all_context)

    # build a chat-style prompt per question
    prompts = [
        [
            # instruction as system prompt
            {'role': 'system', 'content': INSTRUCTIONS},
            # user message containing the question
            {
                'role': 'user', 'content': f'Question: {question} \nDocumentation context: {context} \nYour response:'
            }
        ]
        for question, context in zip(batch['question'], serialized_context)
    ]

    return {
        'prompts': prompts,
        'retrieved_documents': retrieved_documents,
        'retrieved_contexts': retrieved_contexts,
        **batch
    }


def extract_solution(answer: str) -> str:
    '''
    extract the answer
    '''
    # search for the answer marker
    answer_match = re.search(
        r"####\s*Answer:\s*(.*?)(?=\n####\s*Source:|\Z)",
        answer,
        flags=re.IGNORECASE | re.DOTALL,
    )
    # search for the source marker
    source_match = re.search(
        r"####\s*Source:\s*(.*)",
        answer,
        flags=re.IGNORECASE | re.DOTALL,
    )
    answer = answer_match.group(1).strip() if answer_match else answer
    source = source_match.group(1).strip() if source_match else None
    return answer, source


def sample_postprocess_fn(batch: Dict[str, listtype]) -> Dict[str, listtype]:
    '''
    turn raw generated text into a field
    '''
    # extract answer and source from the generated text
    answer, source = [], []
    for text in batch['generated_text']:
        a, s = extract_solution(text)
        answer.append(a)
        source.append(s)
    batch['answer'] = answer
    batch['source'] = source
    return batch


if __name__ == '__main__':
    main()
