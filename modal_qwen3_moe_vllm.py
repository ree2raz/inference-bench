"""Run Qwen3-30B-A3B MoE GPTQ-Int4 reasoning benchmark on vLLM.

Usage: modal run modal_qwen3_moe_vllm.py
"""
import modal
import os
import time

from bench_lib import (
    GPU_TYPE, MODEL_QWEN3_MOE, REASONING_CONCURRENCY_LEVELS, REASONING_REPEATS,
    hf_cache, results_vol,
    make_vllm_image,
    wait_for_server, QWEN3_MOE_VLLM_SERVER_ARGS,
    run_benchmark_impl, print_result_line, write_summary,
)

app = modal.App("inference-bench-qwen3-moe-vllm")


@app.cls(
    image=make_vllm_image(),
    gpu=GPU_TYPE,
    timeout=60 * 180,
    scaledown_window=60,
    volumes={"/hf_cache": hf_cache, "/results": results_vol},
)
class Qwen3MoeVllmBench:
    @modal.enter()
    def start_server(self):
        import subprocess
        self.proc = subprocess.Popen(QWEN3_MOE_VLLM_SERVER_ARGS)
        t0 = time.perf_counter()
        assert wait_for_server(timeout=600), "Qwen3 MoE vLLM server failed to start within 600s"
        self.cold_start = time.perf_counter() - t0
        print(f"[server] Qwen3 MoE vLLM ready in {self.cold_start:.1f}s")

    @modal.method()
    def run_all(self, regimes: list[str]):
        engine = "qwen3_moe_vllm"
        all_results = []
        skipped = 0
        for reg in regimes:
            for conc in REASONING_CONCURRENCY_LEVELS:
                for rep in range(1, REASONING_REPEATS + 1):
                    run_id = f"{engine}_{reg}_c{conc}_r{rep}"
                    remote_path = f"/results/raw/{run_id}.jsonl"
                    if os.path.exists(remote_path):
                        print(f"  [skip] {run_id} already in Volume")
                        skipped += 1
                        continue
                    print(f"\n  >> {engine} | {reg} | c={conc} | r={rep}/{REASONING_REPEATS}")
                    result = run_benchmark_impl(engine, reg, conc, rep, model_name=MODEL_QWEN3_MOE)
                    result["cold_start_seconds"] = round(self.cold_start, 1)
                    print_result_line(result)
                    all_results.append(result)
        if all_results:
            write_summary(all_results)
        if skipped:
            print(f"\n  Skipped {skipped} already-completed configs.")
        return len(all_results), skipped

    @modal.exit()
    def stop_server(self):
        if hasattr(self, "proc") and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=10)


@app.local_entrypoint()
def main():
    regimes = ["reasoning"]
    print(f"=== inference-bench: Qwen3-30B-A3B MoE vLLM ===")
    print(f"  Model:       {MODEL_QWEN3_MOE}")
    print(f"  GPU:         {GPU_TYPE}")
    print(f"  Regimes:     {regimes}")
    bench = Qwen3MoeVllmBench()
    n_new, n_skip = bench.run_all.remote(regimes=regimes)
    print(f"\n  Done: {n_new} new results, {n_skip} skipped.")
