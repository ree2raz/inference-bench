"""Run vLLM AWQ benchmark. Uses Q4-AWQ quantized model.

Usage: modal run modal_vllm_awq.py [--regime short|long]
"""
import modal
import os
import time

from bench_lib import (
    GPU_TYPE, MODEL_AWQ, CONCURRENCY_LEVELS, LONG_CONCURRENCY_LEVELS,
    REPEATS, LONG_REPEATS,
    hf_cache, results_vol,
    make_vllm_awq_image,
    wait_for_server, VLLM_AWQ_SERVER_ARGS,
    run_benchmark_impl, print_result_line, write_summary,
)

app = modal.App("inference-bench-vllm-awq")


def _already_done(engine: str, regime: str, concurrency: int, repeat: int) -> bool:
    fname = f"{engine}_{regime}_c{concurrency}_r{repeat}.jsonl"
    local_path = os.path.join(os.path.dirname(__file__), "results", "raw", fname)
    return os.path.exists(local_path)


@app.cls(
    image=make_vllm_awq_image(),
    gpu=GPU_TYPE,
    timeout=60 * 120,
    scaledown_window=60,
    volumes={"/hf_cache": hf_cache, "/results": results_vol},
)
class VllmAwqBench:
    @modal.enter()
    def start_server(self):
        import subprocess
        self.proc = subprocess.Popen(VLLM_AWQ_SERVER_ARGS)
        t0 = time.perf_counter()
        assert wait_for_server(timeout=300), "vLLM AWQ server failed to start within 300s"
        self.cold_start = time.perf_counter() - t0
        print(f"[server] vLLM AWQ ready in {self.cold_start:.1f}s")

    @modal.method()
    def run_all(self, regimes: list[str]):
        engine = "vllm_awq"
        all_results = []
        skipped = 0
        for reg in regimes:
            conc_levels = CONCURRENCY_LEVELS if reg == "short" else LONG_CONCURRENCY_LEVELS
            n_repeats = REPEATS if reg == "short" else LONG_REPEATS
            for conc in conc_levels:
                for rep in range(1, n_repeats + 1):
                    run_id = f"{engine}_{reg}_c{conc}_r{rep}"
                    remote_path = f"/results/raw/{run_id}.jsonl"
                    if os.path.exists(remote_path):
                        print(f"  [skip] {run_id} already in Volume")
                        skipped += 1
                        continue
                    print(f"\n  >> {engine} | {reg} | c={conc} | r={rep}/{n_repeats}")
                    result = run_benchmark_impl(engine, reg, conc, rep, model_name=MODEL_AWQ)
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
def main(regime: str = "all"):
    regimes = ["short", "long"] if regime == "all" else [regime]

    print(f"=== inference-bench: vLLM AWQ ===")
    print(f"  Model:       {MODEL_AWQ}")
    print(f"  GPU:         {GPU_TYPE}")
    print(f"  Regimes:     {regimes}")

    bench = VllmAwqBench()
    n_new, n_skip = bench.run_all.remote(regimes=regimes)
    print(f"\n  Done: {n_new} new results, {n_skip} skipped.")