"""Shared constants, helpers, and benchmark logic.

Imported by modal_vllm.py, modal_sglang.py, modal_llamacpp.py, and modal_app.py.
No modal.App instance here — each engine file owns its own app.
"""
import modal
import subprocess
import asyncio
import json
import time
import os

GPU_TYPE = "L4"
MODEL_FP16 = "Qwen/Qwen2.5-7B-Instruct"
MODEL_AWQ = "Qwen/Qwen2.5-7B-Instruct-AWQ"
MODEL_GGUF_REPO = "Qwen/Qwen2.5-7B-Instruct-GGUF"
GGUF_FILE = "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
GGUF_FILE_SHARD2 = "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf"

MODEL_QWEN3 = "Qwen/Qwen3-8B"
MODEL_QWEN3_AWQ = "Qwen/Qwen3-8B-AWQ"
MODEL_QWEN3_MOE = "Qwen/Qwen3-30B-A3B-GPTQ-Int4"

CONCURRENCY_LEVELS = [1, 4, 16, 32, 64]
LONG_CONCURRENCY_LEVELS = [1, 16, 64]
REASONING_CONCURRENCY_LEVELS = [1, 4, 16]
TEMPERATURE = 0
MAX_OUTPUT_SHORT = 128
MAX_OUTPUT_LONG = 512
MAX_OUTPUT_REASONING = 8192
REPEATS = 3
LONG_REPEATS = 1
REASONING_REPEATS = 1
SERVER_URL = "http://localhost:8000"

hf_cache = modal.Volume.from_name("inference-bench-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("inference-bench-results", create_if_missing=True)

_WORKLOAD_LOCAL = os.path.join(os.path.dirname(__file__), "prompts", "workload.jsonl")


_SYMLINK_PYTHON = [
    "RUN for p in /usr/local/bin/python3 /usr/bin/python3 /opt/conda/bin/python3; do "
    "if [ -f \"$p\" ]; then ln -sf \"$p\" /usr/local/bin/python && break; fi; done "
    "&& python --version"
]


def make_vllm_image():
    return (
        modal.Image.from_registry(
            "vllm/vllm-openai:v0.8.5",
            setup_dockerfile_commands=_SYMLINK_PYTHON,
        )
        .entrypoint([])
        .pip_install("httpx>=0.27", "numpy>=1.26")
        .env({"HF_HOME": "/hf_cache"})
        .add_local_python_source("bench_lib")
        .add_local_file(_WORKLOAD_LOCAL, remote_path="/opt/prompts/workload.jsonl")
    )


def make_sglang_image():
    return (
        modal.Image.from_registry(
            "lmsysorg/sglang:v0.4.6-cu124",
            setup_dockerfile_commands=_SYMLINK_PYTHON,
        )
        .entrypoint([])
        .pip_install("httpx>=0.27", "numpy>=1.26")
        .env({"HF_HOME": "/hf_cache"})
        .add_local_python_source("bench_lib")
        .add_local_file(_WORKLOAD_LOCAL, remote_path="/opt/prompts/workload.jsonl")
    )


def make_llamacpp_image():
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.1-runtime-ubuntu22.04",
            add_python="3.11",
        )
        .apt_install("libcurl4", "libgomp1")
        .pip_install("huggingface_hub", "httpx>=0.27", "numpy>=1.26")
        .env({"HF_HOME": "/hf_cache", "LD_LIBRARY_PATH": "/hf_cache/llamacpp"})
        .add_local_python_source("bench_lib")
        .add_local_file(_WORKLOAD_LOCAL, remote_path="/opt/prompts/workload.jsonl")
    )


def make_vllm_awq_image():
    return make_vllm_image()


def make_sglang_awq_image():
    return (
        modal.Image.from_registry(
            "lmsysorg/sglang:v0.4.6-cu124",
            setup_dockerfile_commands=_SYMLINK_PYTHON,
        )
        .entrypoint([])
        .pip_install("httpx>=0.27", "numpy>=1.26")
        .run_commands(
            "pip install vllm==0.7.2",
            "pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.5",
        )
        .env({"HF_HOME": "/hf_cache"})
        .add_local_python_source("bench_lib")
        .add_local_file(_WORKLOAD_LOCAL, remote_path="/opt/prompts/workload.jsonl")
    )


def wait_for_server(timeout: int = 300, interval: int = 5):
    import httpx

    waited = 0
    while waited < timeout:
        for endpoint in ["/health", "/v1/models"]:
            try:
                resp = httpx.get(f"{SERVER_URL}{endpoint}", timeout=5)
                if resp.status_code == 200:
                    return True
            except Exception:
                pass
        time.sleep(interval)
        waited += interval
    return False


VLLM_SERVER_ARGS = [
    "python3", "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL_FP16,
    "--host", "0.0.0.0",
    "--port", "8000",
    "--max-num-seqs", "64",
    "--dtype", "auto",
    "--gpu-memory-utilization", "0.90",
    "--disable-log-requests",
]

SGLANG_SERVER_ARGS = [
    "python3", "-m", "sglang.launch_server",
    "--model-path", MODEL_FP16,
    "--host", "0.0.0.0",
    "--port", "8000",
    "--max-running-requests", "64",
    "--mem-fraction-static", "0.85",
    "--disable-cuda-graph",
]

VLLM_AWQ_SERVER_ARGS = [
    "python3", "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL_AWQ,
    "--host", "0.0.0.0",
    "--port", "8000",
    "--max-num-seqs", "64",
    "--quantization", "awq",
    "--enforce-eager",
    "--dtype", "auto",
    "--gpu-memory-utilization", "0.90",
    "--disable-log-requests",
]

SGLANG_AWQ_SERVER_ARGS = [
    "python3", "-m", "sglang.launch_server",
    "--model-path", MODEL_AWQ,
    "--host", "0.0.0.0",
    "--port", "8000",
    "--max-running-requests", "64",
    "--mem-fraction-static", "0.85",
    "--quantization", "awq",
    "--disable-cuda-graph",
]

QWEN3_VLLM_SERVER_ARGS = [
    "python3", "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL_QWEN3,
    "--host", "0.0.0.0",
    "--port", "8000",
    "--max-num-seqs", "32",
    "--dtype", "auto",
    "--gpu-memory-utilization", "0.90",
    "--enable-reasoning",
    "--reasoning-parser", "deepseek_r1",
    "--disable-log-requests",
]

QWEN3_SGLANG_SERVER_ARGS = [
    "python3", "-m", "sglang.launch_server",
    "--model-path", MODEL_QWEN3,
    "--host", "0.0.0.0",
    "--port", "8000",
    "--max-running-requests", "32",
    "--mem-fraction-static", "0.85",
    "--reasoning-parser", "deepseek-r1",
    "--disable-cuda-graph",
]

QWEN3_AWQ_VLLM_SERVER_ARGS = [
    "python3", "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL_QWEN3_AWQ,
    "--host", "0.0.0.0",
    "--port", "8000",
    "--max-num-seqs", "32",
    "--quantization", "awq",
    "--enforce-eager",
    "--dtype", "auto",
    "--gpu-memory-utilization", "0.90",
    "--enable-reasoning",
    "--reasoning-parser", "deepseek_r1",
    "--disable-log-requests",
]

QWEN3_AWQ_SGLANG_SERVER_ARGS = [
    "python3", "-m", "sglang.launch_server",
    "--model-path", MODEL_QWEN3_AWQ,
    "--host", "0.0.0.0",
    "--port", "8000",
    "--max-running-requests", "32",
    "--mem-fraction-static", "0.85",
    "--quantization", "awq",
    "--reasoning-parser", "deepseek-r1",
    "--disable-cuda-graph",
    "--presence-penalty", "1.5",
]

QWEN3_MOE_VLLM_SERVER_ARGS = [
    "python3", "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL_QWEN3_MOE,
    "--host", "0.0.0.0",
    "--port", "8000",
    "--max-num-seqs", "16",
    "--max-model-len", "4096",
    "--quantization", "gptq",
    "--dtype", "auto",
    "--gpu-memory-utilization", "0.90",
    "--enable-reasoning",
    "--reasoning-parser", "deepseek_r1",
    "--disable-log-requests",
]

QWEN3_MOE_SGLANG_SERVER_ARGS = [
    "python3", "-m", "sglang.launch_server",
    "--model-path", MODEL_QWEN3_MOE,
    "--host", "0.0.0.0",
    "--port", "8000",
    "--max-running-requests", "16",
    "--mem-fraction-static", "0.85",
    "--quantization", "gptq",
    "--max-model-len", "4096",
    "--reasoning-parser", "deepseek-r1",
    "--disable-cuda-graph",
]


def start_llamacpp_server():
    import stat
    server_bin = "/hf_cache/llamacpp/llama-server"
    st = os.stat(server_bin)
    if not (st.st_mode & stat.S_IEXEC):
        os.chmod(server_bin, st.st_mode | stat.S_IEXEC)

    from huggingface_hub import hf_hub_download

    gguf_path = hf_hub_download(
        repo_id=MODEL_GGUF_REPO,
        filename=GGUF_FILE,
        cache_dir="/hf_cache",
    )
    hf_hub_download(
        repo_id=MODEL_GGUF_REPO,
        filename=GGUF_FILE_SHARD2,
        cache_dir="/hf_cache",
    )
    return subprocess.Popen([
        server_bin,
        "-m", gguf_path,
        "--host", "0.0.0.0",
        "--port", "8000",
        "-np", "4",
        "-ngl", "99",
        "--parallel", "4",
        "-c", "16384",
    ])


def run_benchmark_impl(
    engine: str,
    regime: str,
    concurrency: int,
    repeat: int,
    model_name: str | None = None,
) -> dict:
    import httpx
    import numpy as np

    workload_path = "/opt/prompts/workload.jsonl"
    prompts = []
    with open(workload_path) as f:
        for line in f:
            p = json.loads(line)
            if p["regime"] == regime:
                prompts.append(p)

    max_tokens = MAX_OUTPUT_SHORT if regime == "short" else (MAX_OUTPUT_LONG if regime == "long" else MAX_OUTPUT_REASONING)
    warmup_prompts = prompts[:concurrency * 2]
    measure_prompts = prompts[:max(concurrency * 30, len(prompts))]
    if model_name is None:
        model_name = "default" if engine == "llamacpp" else MODEL_FP16
    supports_stream_usage = engine != "llamacpp"
    is_reasoning = regime == "reasoning"
    measure_count = min(len(measure_prompts), 10 if is_reasoning else len(measure_prompts))
    measure_prompts = measure_prompts[:measure_count]

    async def _send_streaming(client: httpx.AsyncClient, prompt: dict) -> dict:
        payload = {
            "model": model_name,
            "messages": prompt["messages"],
            "max_tokens": max_tokens,
            "temperature": TEMPERATURE,
            "stream": True,
        }
        if supports_stream_usage:
            payload["stream_options"] = {"include_usage": True}

        t_start = time.perf_counter()
        first_token_time = None
        first_answer_time = None
        output_tokens = 0
        reasoning_tokens = 0
        answer_tokens = 0
        try:
            async with client.stream(
                "POST",
                f"{SERVER_URL}/v1/chat/completions",
                json=payload,
                timeout=600,
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    t_end = time.perf_counter()
                    return {
                        "success": False,
                        "error": f"HTTP {resp.status_code}: {body[:200]}",
                        "wall_time": t_end - t_start,
                    }
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        if chunk.get("choices"):
                            delta = chunk["choices"][0].get("delta", {})
                            reasoning_content = delta.get("reasoning_content", "")
                            content = delta.get("content", "")
                            if reasoning_content and first_token_time is None:
                                first_token_time = time.perf_counter()
                                reasoning_tokens += 1
                            elif reasoning_content:
                                reasoning_tokens += 1
                            if content and first_token_time is None:
                                first_token_time = time.perf_counter()
                                answer_tokens += 1
                            elif content:
                                if first_answer_time is None:
                                    first_answer_time = time.perf_counter()
                                answer_tokens += 1
                        usage = chunk.get("usage")
                        if usage:
                            output_tokens = usage.get("completion_tokens", 0)
                            if usage.get("completion_tokens_details"):
                                cd = usage["completion_tokens_details"]
                                reasoning_tokens = cd.get("reasoning_tokens", reasoning_tokens)
                    except json.JSONDecodeError:
                        pass
            t_end = time.perf_counter()
            ttft = (first_token_time - t_start) if first_token_time else None
            ttft_answer = (first_answer_time - t_start) if first_answer_time else None
            if answer_tokens == 0 and reasoning_tokens > 0:
                answer_tokens = output_tokens - reasoning_tokens if output_tokens > reasoning_tokens else 0
            elif output_tokens > 0 and reasoning_tokens == 0 and answer_tokens > 0:
                reasoning_tokens = max(0, output_tokens - answer_tokens)
            return {
                "success": True,
                "wall_time": t_end - t_start,
                "ttft": ttft,
                "ttft_answer": ttft_answer,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
                "answer_tokens": answer_tokens,
            }
        except Exception as e:
            t_end = time.perf_counter()
            return {"success": False, "error": str(e), "wall_time": t_end - t_start}

    async def _send_non_streaming(client: httpx.AsyncClient, prompt: dict) -> dict:
        payload = {
            "model": model_name,
            "messages": prompt["messages"],
            "max_tokens": max_tokens,
            "temperature": TEMPERATURE,
        }
        t_start = time.perf_counter()
        try:
            resp = await client.post(
                f"{SERVER_URL}/v1/chat/completions",
                json=payload,
                timeout=300,
            )
            t_end = time.perf_counter()
            if resp.status_code != 200:
                return {
                    "success": False,
                    "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                    "wall_time": t_end - t_start,
                }
            body = resp.json()
            usage = body.get("usage", {})
            output_tokens = usage.get("completion_tokens", 0)
            return {
                "success": True,
                "wall_time": t_end - t_start,
                "ttft": None,
                "output_tokens": output_tokens,
            }
        except Exception as e:
            t_end = time.perf_counter()
            return {"success": False, "error": str(e), "wall_time": t_end - t_start}

    async def _run_batch(prompts_batch: list[dict]) -> list[dict]:
        sem = asyncio.Semaphore(concurrency)

        async def _limited(p):
            async with sem:
                async with httpx.AsyncClient() as client:
                    if supports_stream_usage:
                        return await _send_streaming(client, p)
                    else:
                        return await _send_non_streaming(client, p)

        return list(await asyncio.gather(*[_limited(p) for p in prompts_batch]))

    print(f"[warmup] {engine} {regime} c={concurrency} r={repeat}")
    asyncio.run(_run_batch(warmup_prompts))
    print(f"[warmup done]")

    print(f"[measure] {len(measure_prompts)} requests at concurrency={concurrency} (streaming={supports_stream_usage})")
    t0 = time.perf_counter()
    results = asyncio.run(_run_batch(measure_prompts))
    wall_seconds = time.perf_counter() - t0

    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    if not successful:
        return {"error": "all requests failed", "failures": [str(f) for f in failed[:5]]}

    output_tokens_list = [r["output_tokens"] for r in successful if r.get("output_tokens")]
    ttft_list = [r["ttft"] for r in successful if r.get("ttft") is not None]
    ttft_answer_list = [r["ttft_answer"] for r in successful if r.get("ttft_answer") is not None]
    reasoning_tokens_list = [r.get("reasoning_tokens", 0) for r in successful]
    answer_tokens_list = [r.get("answer_tokens", 0) for r in successful]
    wall_list = [r["wall_time"] for r in successful]
    total_output_tokens = sum(output_tokens_list)
    throughput = total_output_tokens / wall_seconds if wall_seconds > 0 else 0

    def pct(data, p):
        if not data:
            return None
        s = sorted(data)
        idx = (p / 100) * (len(s) - 1)
        lo = int(idx)
        hi = min(lo + 1, len(s) - 1)
        frac = idx - lo
        return s[lo] * (1 - frac) + s[hi] * frac

    tpot_list = []
    for r in successful:
        if r.get("ttft") and r.get("output_tokens", 0) > 1:
            tpot_list.append((r["wall_time"] - r["ttft"]) / (r["output_tokens"] - 1))

    result = {
        "engine": engine,
        "regime": regime,
        "concurrency": concurrency,
        "repeat": repeat,
        "total_requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "wall_seconds": round(wall_seconds, 3),
        "total_output_tokens": total_output_tokens,
        "throughput_tokens_per_sec": round(throughput, 2),
        "throughput_per_request": round(throughput / concurrency, 2) if concurrency else 0,
        "ttft_p50": round(pct(ttft_list, 50), 4) if ttft_list else None,
        "ttft_p95": round(pct(ttft_list, 95), 4) if ttft_list else None,
        "tpot_p50": round(pct(tpot_list, 50), 4) if tpot_list else None,
        "tpot_p95": round(pct(tpot_list, 95), 4) if tpot_list else None,
        "latency_p50": round(pct(wall_list, 50), 4),
        "latency_p95": round(pct(wall_list, 95), 4),
        "latency_p99": round(pct(wall_list, 99), 4),
        "output_tokens_mean": round(float(np.mean(output_tokens_list)), 1) if output_tokens_list else 0,
        "reasoning_tokens_mean": round(float(np.mean(reasoning_tokens_list)), 1) if reasoning_tokens_list else 0,
        "answer_tokens_mean": round(float(np.mean(answer_tokens_list)), 1) if answer_tokens_list else 0,
        "ttft_answer_p50": round(pct(ttft_answer_list, 50), 4) if ttft_answer_list else None,
    }

    run_id = f"{engine}_{regime}_c{concurrency}_r{repeat}"
    raw_path = f"/results/raw/{run_id}.jsonl"
    os.makedirs("/results/raw", exist_ok=True)
    with open(raw_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    results_vol.commit()

    return result


def print_result_line(result: dict):
    if "error" in result and "engine" not in result:
        print(f"     ERROR: {result['error']}")
    else:
        tp = result.get("throughput_tokens_per_sec", 0)
        ttft = result.get("ttft_p50", "N/A")
        lat = result.get("latency_p95", "N/A")
        ok = result.get("successful_requests", 0)
        total = result.get("total_requests", 0)
        print(f"     throughput={tp:.1f} tok/s  ttft_p50={ttft}  lat_p95={lat}  ok={ok}/{total}")


def write_summary(results: list):
    summary_path = os.path.join(os.path.dirname(__file__), "results", "summary.jsonl")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\n  {len(results)} configs -> results/summary.jsonl")
