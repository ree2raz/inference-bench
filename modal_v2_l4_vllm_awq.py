"""v2 sweep: vLLM v0.20.1 AWQ on L4. Mirrors modal_vllm_awq.py.

Engine name: vllm_v2 with model_name=MODEL_AWQ. Result tag: vllm_v2_awq_*.
Usage: modal run --detach modal_v2_l4_vllm_awq.py
"""
import modal
import os
import subprocess
import time
import json

from bench_lib import (
    GPU_TYPE, MODEL_AWQ, CONCURRENCY_LEVELS, REPEATS,
    hf_cache, results_vol,
    make_vllm_v2_image,
    wait_for_server, VLLM_V2_AWQ_SERVER_ARGS,
    run_benchmark_impl, print_result_line,
)

ENGINE = "vllm_v2_awq"
app = modal.App("inference-bench-v2-l4-vllm-awq")


@app.cls(
    image=make_vllm_v2_image(),
    gpu=GPU_TYPE,
    timeout=60 * 120,
    scaledown_window=60,
    volumes={"/hf_cache": hf_cache, "/results": results_vol},
)
class V2VllmL4AwqBench:
    @modal.enter()
    def start_server(self):
        self.proc = subprocess.Popen(VLLM_V2_AWQ_SERVER_ARGS)
        t0 = time.perf_counter()
        assert wait_for_server(timeout=300), "vLLM AWQ server failed to start within 300s"
        self.cold_start = time.perf_counter() - t0
        print(f"[server] vLLM v0.20.1 AWQ ready in {self.cold_start:.1f}s")

    @modal.method()
    def run(self, concurrency: int, repeat: int) -> dict:
        run_id = f"{ENGINE}_short_c{concurrency}_r{repeat}"
        remote_path = f"/results/raw/{run_id}.jsonl"
        if os.path.exists(remote_path):
            return {"skipped": True}
        result = run_benchmark_impl(ENGINE, "short", concurrency, repeat, model_name=MODEL_AWQ)
        result["cold_start_seconds"] = round(self.cold_start, 1)
        result["engine_version"] = "vllm-v0.20.1"
        result["quantization"] = "awq"
        return result

    @modal.exit()
    def stop_server(self):
        if hasattr(self, "proc") and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=10)


@app.local_entrypoint()
def main():
    print(f"=== v2 sweep: {ENGINE} L4 AWQ ===")
    bench = V2VllmL4AwqBench()
    all_results = []
    for conc in CONCURRENCY_LEVELS:
        for rep in range(1, REPEATS + 1):
            print(f"\n  >> {ENGINE} | short | c={conc} | r={rep}/{REPEATS}")
            r = bench.run.remote(concurrency=conc, repeat=rep)
            if not r.get("skipped"):
                print_result_line(r)
                all_results.append(r)
    if all_results:
        outpath = os.path.join(os.path.dirname(__file__), "results", "v2_l4_vllm_awq.jsonl")
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        with open(outpath, "w") as f:
            for r in all_results:
                f.write(json.dumps(r) + "\n")
        print(f"\n  {len(all_results)} configs -> {outpath}")
