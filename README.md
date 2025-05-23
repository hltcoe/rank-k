# Rank-K

Details TBA. Please come back later or raise an issue if you would like to obtain more details sooner.  

Model: https://huggingface.co/hltcoe/Rank-K-32B

Paper: https://arxiv.org/abs/2505.14432

## Get Started

The following commands will create an enviornment for running Rank-K. 

```bash
conda create -n rankk python=3.12
conda activate rankk
pip install uv # uv makes things super fast! 
uv pip install vllm transformers datasets ir_datasets pandas tqdm openai
```

## Usage 

The following example will rerank the `theoremqa_theorems` subset based on a BM25 run provided by Rank1. 
You can download the run file [here](https://huggingface.co/datasets/jhu-clsp/rank1-run-files/resolve/main/theoremqa_theorems_bm25_long_False/score.json). 
Assuming you name this BM25 score file as `first_stage_score.json`, the following command will rerank the top 100 documents from BM25 
using a sliding window of size 20 and a stride of 10 using 4 GPUs and 16 concurrent workers. 

```bash
VLLM_SKIP_P2P_CHECK=1 python rerank.py \
--rerank_data first_stage_score.json \
--dataset_revision a75a0eb \
--docs xlangai/BRIGHT:theoremqa_theorems \
--queries xlangai/BRIGHT:theoremqa_theorems \
--truncate_doc_to 450 \
--rerank_depth 100 \
--rerank_window 20 \
--rerank_stride 10 \
--output test_output.json \
--gpus 4 \
--worker 16 \
--temperature 0.5 \
--max-token 8000
```

Note that in some enviornment, you might need to set the environment variable `VLLM_SKIP_P2P_CHECK=1` to ensure VLLM does 
not try to allocate more than one process to a cuda device when using multiple GPUs. 

## Reference

Please cite the following paper if you use the model. 

```bibtex
@article{rank-k,
    title={Rank-K: Test-Time Reasoning for Listwise Reranking},
    author={Yang, Eugene and Yates, Andrew and Ricci, Kathryn and Weller, Orion and Chari, Vivek and Van Durme, Benjamin and Lawrie, Dawn},
    journal={arXiv preprint arXiv:2505.14432},
    year={2025}
}
```