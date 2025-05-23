from pathlib import Path

import pandas as pd
import json

import ir_datasets as irds
from datasets import load_dataset

from tqdm import tqdm

def safe_load_json(fn):
    for line in open(fn):
        try:
            yield json.loads(line)
        except json.decoder.JSONDecodeError:
            continue

class _irds_wrapper:
    def __init__(self, ds):
        self.ds = ds
    
    def __getitem__(self, idx):
        return self.ds.docs.lookup(idx).default_text()


def load_data(args):
    ds = irds.load(args.dataset_id)
    if args.docs is None:
        docs = _irds_wrapper(ds)
    else:
        if args.docs.endswith(".jsonl"):
            docs = {
                line['id']: line['title'] + ' ' + line['text']
                for line in tqdm(safe_load_json(args.docs), desc='load docs')
            }
        elif args.docs.startswith('xlangai/BRIGHT'):
            subset = args.docs.split(":")[1]
            docs = {
                doc['id']: doc['content']
                for doc in load_dataset(
                    'xlangai/BRIGHT', 'documents', revision=args.dataset_revision
                )[subset]
            }
        else:
            docs = {
                line[0]: line[1]
                for line in tqdm(map(lambda x: x.strip().split("\t"), open(args.docs)))
            }

    exclusion = None
    if args.queries is None:
        queries = { q.query_id: q.default_text() for q in ds.queries }
    elif args.queries.startswith('xlangai/BRIGHT'):
        subset = args.queries.split(":")[1]
        subset_ds = load_dataset(
            'xlangai/BRIGHT', 'examples', revision=args.dataset_revision
        )[subset]
        queries = {
            ex['id']: ex['query']
            for ex in subset_ds
        }
        exclusion = { ex['id']: set(ex['excluded_ids']) - set(['N/A']) for ex in subset_ds }
    else:
        queries = dict(map(lambda x: x.strip().split("\t"), open(args.queries)))

    return docs, exclusion, queries


def load_init_run(args, queries, exclusion):
    
    if args.rerank_data.endswith('.json'):
        return pd.DataFrame([
            { 'query_id': qid, 'doc_id': did, 'score': score }
            for qid, ranking in json.loads(Path(args.rerank_data).read_text()).items()
            for did, score in sorted(ranking.items(), key=lambda x: -x[1]) 
            if qid in queries
            if exclusion is None or did not in exclusion[qid]
        ])
    else:
        return pd.DataFrame([  
            {
                'query_id': line[0],
                'doc_id': line[2],
                'score': float(line[4])
            }
            for line in tqdm(map(lambda x: x.split(), open(args.rerank_data))) 
        ])


def combine_passages(passages):
    return "\n\n".join(
        f"[{i+1}] {text}" for i, text in enumerate(passages)
    )
