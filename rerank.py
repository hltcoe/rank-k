import os
import sys
import argparse
import json
from openai import OpenAI
from tqdm import tqdm

from transformers import AutoTokenizer

from launch_vllm import launch_vllm
from multiprocessing import Process, Pool

from utils import load_data, load_init_run, combine_passages

rank_k_prompt = """
Determine a ranking of the passages based on how relevant they are to the query. 
If the query is a question, how relevant a passage is depends on how well it answers the question. 
If not, try analyze the intent of the query and assess how well each passage satisfy the intent. 
The query may have typos and passages may contain contradicting information. 
However, we do not get into fact-checking. We just rank the passages based on they relevancy to the query. 

Sort them from the most relevant to the least. 
Answer with the passage number using a format of `[3] > [2] > [4] = [1] > [5]`. 
Ties are acceptable if they are equally relevant. 
I need you to be accurate but overthinking it is unnecessary.
Output only the ordering without any other text.

Query: {query}

{docs}
"""

def parse_ranking_more(rank_string, doc_ids):
    rr = rank_string.split(">")
    
    scores = {}
    for i, p in enumerate(rr):
        for pidstring in p.split("="):
            pid = pidstring.strip().replace("[", "").replace("]", "")
            
            try: 
                pid = int(pid)
            except: 
                # print(pid)
                continue
            try:
                did = doc_ids[ pid-1 ]
            except IndexError:
                # print(pidstring, pid)
                continue
                
            if did in scores:
                return scores
            scores[ did ] = 1/(i+1)
        
    return scores

def rerank(query, doc_ids, candidates, client: OpenAI, args):
    content = rank_k_prompt.format(
        query=query,
        docs=combine_passages(candidates)
    )

    resp = client.chat.completions.create(
        model=args.model,
        temperature=args.temperature,
        timeout=5400,
        max_tokens=args.max_tokens,
        messages=[
            {"role": "user", "content": content}
        ]
    )

    scores = parse_ranking_more(
        resp.choices[0].message.content.strip().split("\n")[-1],
        doc_ids
    )

    new_ranking = [
        doc_id for doc_id, score in sorted(scores.items(), key=lambda x: -x[1])
    ]

    if len(new_ranking) < len(doc_ids):
        new_ranking += [
            d for d in doc_ids if d not in new_ranking
        ]
    
    return new_ranking

def rerank_topic(worker_args):
    args, qid, query, doc_subset, to_rerank_doc_ids = worker_args

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        base_url=f"http://{args.server}:{args.port}/v1",
    )

    if args.truncate_doc_to is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        doc_subset = {
            doc_id: tokenizer.decode(tokenizer.encode(d)[:args.truncate_doc_to])
            for doc_id, d in doc_subset.items()
        }

    for rerank_end_idx in range(args.rerank_depth, 0, -args.rerank_stride):
        rerank_start_idx = rerank_end_idx - args.rerank_window
        
        doc_ids = to_rerank_doc_ids[rerank_start_idx:rerank_end_idx].copy()

        to_rerank_doc_ids[rerank_start_idx:rerank_end_idx] = rerank(
            query, doc_ids, [ doc_subset[did] for did in doc_ids ], 
            client, args
        )

        if rerank_start_idx == 0: 
            # happens to rerank the very first one
            break
    
    return qid, to_rerank_doc_ids

def rerank_all_topics(args):
    
    docs, exclusion, queries = load_data(args)
    all_records = load_init_run(args, queries, exclusion)

    def yield_task():
        for qid, ranklist in all_records.groupby('query_id', sort=True):
            to_rerank_doc_ids = ranklist.sort_values('score', ascending=False)\
                                .iloc[:args.rerank_depth].doc_id.tolist()
            yield (
                args, 
                qid, 
                queries[qid], 
                { doc_id: docs[doc_id] for doc_id in to_rerank_doc_ids }, 
                to_rerank_doc_ids
            )

    jobs = yield_task()

    raw_output = {}

    assert args.worker >= 1
    if args.worker == 1:
        for qid, worker_output in tqdm(map(rerank_topic, jobs), desc="reranking topics"):
            raw_output[qid] = worker_output
    else:
        with Pool(args.worker) as p:
            runner = tqdm(
                p.imap_unordered(rerank_topic, jobs), 
                total=all_records['query_id'].unique().size, 
                desc="reranking topics",
                dynamic_ncols=True
            )
            for qid, worker_output in runner:
                raw_output[qid] = worker_output

    with open(args.output, 'w') as fw:
        json.dump({
            qid: {
                doc_id: 1/(i+1)
                for i, doc_id in enumerate(ranking)
            }
            for qid, ranking in raw_output.items()
        }, fw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--rerank_data', required=True, 
                        help="Rank list to be reranked in json")
    parser.add_argument('--dataset_id', type=str, default='msmarco-passage/train', 
                        help="ir_dataset dataset id for rerank; "
                             "ignored when --docs and --queries are set")
    
    parser.add_argument('--docs', type=str, default=None,
                        help="HF datasets for documents to rerank")
    parser.add_argument('--queries', type=str, default=None,
                        help="HF datasets for queries to rerank")
    parser.add_argument('--dataset_revision', type=str, default=None,
                        help="HF datasets revision tag for --docs and --queries")

    parser.add_argument('--truncate_doc_to', type=int, default=None,
                        help="truncate passage to # of tokens at reranking")

    parser.add_argument('--rerank_depth', type=int, default=100,
                        help="number of top-ranked documents to rerank")
    parser.add_argument('--rerank_window', type=int, default=20,
                        help="number of documents in each reranking call")
    parser.add_argument('--rerank_stride', type=int, default=10,
                        help="stride of moving window of documents during reranking")

    parser.add_argument('--gpus', type=int, default=1,
                        help="number of GPUs for running LM")
    parser.add_argument('--model', type=str, default="hltcoe/Rank-K-32B", 
                        help="Name or path of the model to use")
    parser.add_argument('--temperature', type=float, default=0.7,
                        help="temperature for generation")
    parser.add_argument('--max-tokens', type=int, default=4000,
                        help="maximum tokens for genreration")
    
    parser.add_argument('--server', default=None,
                        help="OpenAI-compatible server running Rank-K model; "
                             "will launch the model in back if not provided")
    parser.add_argument('--port', default=8000,
                        help="port of the api server")

    parser.add_argument('--worker', type=int, default=1,
                        help="number of concurrent workers for running Rank-K")

    parser.add_argument('--output', required=True, 
                        help="Path to save the output")

    args = parser.parse_args()

    # borrowed from 
    # https://github.com/hltcoe/llm-heapsort-reranking/blob/main/llm_heapsort_reranking/run.py
    vllm_process = None
    if args.server is None:
        vllm_process = launch_vllm(args.model, args.gpus, port=args.port)
        args.server = "localhost"

    if vllm_process:
        retcode = 1
        try:
            p = Process(target=rerank_all_topics, args=(args,))
            p.start()
            p.join()
            vllm_process.terminate()
            retcode = 0
        finally:
            sys.exit(retcode)
    else:
        rerank_all_topics(args)
