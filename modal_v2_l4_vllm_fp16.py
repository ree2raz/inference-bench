"""v2 sweep: vLLM v0.20.1 dense FP16 on L4. Mirrors modal_vllm.py.

Engine name: vllm_v2 (results saved to /results/raw/vllm_v2_*.jsonl).
Usage: modal run --detach modal_v2_l4_vllm_fp16.py
"""
import modal
import os
import subprocess
import time

from bench_lib import (
    GPU_TYPE, MODEL_FP16, CONCURRENCY_LEVELS, LONG_CONCURRENCY_LEVELS,
    REPEATS, LONG_REPEATS,
    hf_cache, results_vol,
    make_vllm_v2_image,
    wait_for_server, VLLM_V2_SERVER_ARGS,
    run_benchmark_impl, print_result_line, write_summary,
)

ENGINE = "vllm_v2"
app = modal.App("inference-bench-v2-l4-vllm-fp16")


@app.cls(
    image=make_vllm_v2_image(),
    gpu=GPU_TYPE,
    timeout=60 * 120,
    scaledown_window=60,
    volumes={"/hf_cache": hf_cache, "/results": results_vol},
)
class V2VllmL4Bench:
    @modal.enter()
    def start_server(self):
        self.proc = subprocess.Popen(VLLM_V2_SERVER_ARGS)
        t0 = time.perf_counter()
        assert wait_for_server(timeout=300), "vLLM server failed to start within 300s"
        self.cold_start = time.perf_counter() - t0
        print(f"[server] vLLM v0.20.1 ready in {self.cold_start:.1f}s")

    @modal.method()
    def run(self, regime: str, concurrency: int, repeat: int) -> dict:
        run_id = f"{ENGINE}_{regime}_c{concurrency}_r{repeat}"
        remote_path = f"/results/raw/{run_id}.jsonl"
        if os.path.exists(remote_path):
            print(f"[skip] {run_id} already in Volume")
            return {"skipped": True, "engine": ENGINE, "regime": regime, "concurrency": concurrency, "repeat": repeat}
        result = run_benchmark_impl(ENGINE, regime, concurrency, repeat)
        result["cold_start_seconds"] = round(self.cold_start, 1)
        result["engine_version"] = "vllm-v0.20.1"
        return result

    @modal.exit()
    def stop_server(self):
        if hasattr(self, "proc") and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=10)


@app.local_entrypoint()
def main(regime: str = "all"):
    regimes = ["short", "long"] if regime == "all" else [regime]
    print(f"=== v2 sweep: {ENGINE} L4 FP16 ===")
    print(f"  Image:       vllm/vllm-openai:v0.20.1")
    print(f"  Model:       {MODEL_FP16}")
    print(f"  GPU:         {GPU_TYPE}")
    print(f"  Regimes:     {regimes}")

    bench = V2VllmL4Bench()
    all_results = []
    for reg in regimes:
        conc_levels = CONCURRENCY_LEVELS if reg == "short" else LONG_CONCURRENCY_LEVELS
        n_repeats = REPEATS if reg == "short" else LONG_REPEATS
        for conc in conc_levels:
            for rep in range(1, n_repeats + 1):
                print(f"\n  >> {ENGINE} | {reg} | c={conc} | r={rep}/{n_repeats}")
                result = bench.run.remote(regime=reg, concurrency=conc, repeat=rep)
                if not result.get("skipped"):
                    print_result_line(result)
                    all_results.append(result)

    if all_results:
        outpath = os.path.join(os.path.dirname(__file__), "results", "v2_l4_vllm_fp16.jsonl")
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        import json
        with open(outpath, "w") as f:
            for r in all_results:
                f.write(json.dumps(r) + "\n")
        print(f"\n  {len(all_results)} configs -> {outpath}")
