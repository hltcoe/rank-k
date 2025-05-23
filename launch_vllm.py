# Adapted from https://github.com/hltcoe/llm-heapsort-reranking/blob/main/llm_heapsort_reranking/vllm.py

import subprocess
import sys

def launch_vllm(model: str, gpus: int, port: int = 25251):
    cmd = f"vllm serve {model} --gpu-memory-utilization 0.95 --tensor-parallel-size {gpus} --port {port}"
    server_process = subprocess.Popen(
        cmd.split(),
        stderr=subprocess.PIPE,
        text=True
    )
    
    while True:
        line = server_process.stderr.readline()
        sys.stdout.write(line)
        if 'Application startup complete.' in line.strip():
            print(f"====== vLLM server is ready!")
            return server_process