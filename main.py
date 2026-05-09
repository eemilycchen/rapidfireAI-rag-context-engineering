import argparse
from pathlib import Path
import json
from datasets import Dataset

from typing import Dict
from typing import List as listtype
import re

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

TRITON_API_KEY = Path('./api-key.txt').expanduser().read_text(encoding='utf-8').splitlines()[0].strip()
BASE_URL = 'https://tritonai-api.ucsd.edu'
DATA_DIR = './data/sourcedocs/'
GENERATION_MODEL = 'api-gpt-oss-120b'


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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to input JSON')
    parser.add_argument('--output', required=True, help='Path to output JSON')
    parser.add_argument('--corpus-dir', required=False, help='Path to the directiory containing the corpus', default=DATA_DIR)
    parser.add_argument('--apikey-txt', required=False, help='API key for the LLM', default='~/api-key.txt')
    parser.add_argument('--generation-model', required=False, help='LLM model used for RAG', default=GENERATION_MODEL)
    
    args = parser.parse_args()

    TRITON_API_KEY = Path(args.apikey_txt).expanduser().read_text(encoding='utf-8').splitlines()[0].strip()

    # define document loader
    loader = DirectoryLoader(
        path=args.corpus_dir, glob='**/*.rst', loader_cls=TextLoaderWithMetadata,
        loader_kwargs={
            'metadata_func': metadata_func,
        },
        show_progress=True, use_multithreading=True
    )

    # define text splitter
    text_splitter = TextSplitterWithLineNum().from_tiktoken_encoder(
        encoding_name='o200k_base', chunk_size=384, chunk_overlap=64, separators=['\n', '\n\n']
    )

    # define embedding configurations
    embedding_cfg = {
        'class': OpenAIEmbeddings,
        'model': 'api-tgpt-embeddings',
        'api_key': TRITON_API_KEY,
        'base_url': BASE_URL,
        'check_embedding_ctx_length': False
    }

    # define vector storage configurations
    vector_store_cfg={'type': 'faiss', 'batch_size': batch_size}

    # define searching algorithm configurations
    search_cfg=List([{'type': 'mmr', 'k': 15}])

    # define reranker configurations
    reranker_cfg={
        'class': CrossEncoderReranker, 'model_name': 'BAAI/bge-reranker-v2-m3',
        'model_kwargs': {'device': 'cpu'}, 'top_n': 8,
    }

    # define rag module
    rag = RFLangChainRagSpec(
        # to load the corpus
        document_loader=loader,
        # to split documents into chunks
        text_splitter=text_splitter,
        # to embed the chunks
        embedding_cfg=embedding_cfg,
        # vectore storage
        vector_store_cfg=vector_store_cfg,
        # search algorithm
        search_cfg=search_cfg,
        # rerank top results with a cross encoder
        reranker_cfg=reranker_cfg,
        # whether to use gpu-accelerated faiss
        enable_gpu_search=False,
        # to format each retrieved chunk for context injection into prompts
        document_template=custom_template,
    )

    # open ai model configurations
    openai_config = RFOpenAIAPIModelConfig(
        client_config={'api_key': TRITON_API_KEY, 'base_url': BASE_URL, 'max_retries': 2},
        model_config={
            'model': args.generation_model,
            'max_completion_tokens': 2048,
            # # two different reasoning levels
            'reasoning_effort': 'high'
        },
        # request per minute needs to be set based on your account tier and the specific model used
        rpm_limit=120,
        # token per minute needs to be set based on your account tier and the specific model used
        tpm_limit=1_000_000,
        # rag model
        rag=rag,
        prompt_manager=None
    )

    # set up a grid search for all the configurations
    config_set = List([{
        'openai_config': List([
            openai_config
        ]),
        'batch_size': batch_size,
        'preprocess_fn': sample_preprocess_fn, 'postprocess_fn': sample_postprocess_fn,
        'compute_metrics_fn': {}, 'accumulate_metrics_fn': {},
    }])
    config_group = RFGridSearch(config_set)

    # create experiments
    experiment = Experiment(experiment_name='rag-generation-only', mode='evals')

    # load the questions
    questions = load_question_set(args.input)

    # run the experiments
    results = experiment.run_evals(config_group=config_group, dataset=questions, num_shards=4, seed=42)
    experiment.end()

    print('experiments finished')

    # write the results to the output json file
    write_answer(args.output, results)


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