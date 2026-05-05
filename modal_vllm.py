"""Run vLLM benchmark only. No other engine images are built.

Usage: modal run modal_vllm.py [--regime short|long]
"""
import modal
import os
import subprocess
import time

from bench_lib import (
    GPU_TYPE, MODEL_FP16, CONCURRENCY_LEVELS, LONG_CONCURRENCY_LEVELS,
    REPEATS, LONG_REPEATS,
    hf_cache, results_vol,
    make_vllm_image,
    wait_for_server, VLLM_SERVER_ARGS,
    run_benchmark_impl, print_result_line, write_summary,
)

app = modal.App("inference-bench-vllm")


def _already_done(engine: str, regime: str, concurrency: int, repeat: int) -> bool:
    fname = f"{engine}_{regime}_c{concurrency}_r{repeat}.jsonl"
    local_path = os.path.join(os.path.dirname(__file__), "results", "raw", fname)
    if os.path.exists(local_path):
        return True
    remote_path = f"/results/raw/{fname}"
    if os.path.exists(remote_path):
        return True
    return False


@app.cls(
    image=make_vllm_image(),
    gpu=GPU_TYPE,
    timeout=60 * 120,
    scaledown_window=60,
    volumes={"/hf_cache": hf_cache, "/results": results_vol},
)
class VllmBench:
    @modal.enter()
    def start_server(self):
        self.proc = subprocess.Popen(VLLM_SERVER_ARGS)
        t0 = time.perf_counter()
        assert wait_for_server(timeout=300), "vLLM server failed to start within 300s"
        self.cold_start = time.perf_counter() - t0
        print(f"[server] vLLM ready in {self.cold_start:.1f}s")

    @modal.method()
    def run(self, regime: str, concurrency: int, repeat: int) -> dict:
        run_id = f"vllm_{regime}_c{concurrency}_r{repeat}"
        remote_path = f"/results/raw/{run_id}.jsonl"
        if os.path.exists(remote_path):
            print(f"[skip] {run_id} already in Volume")
            return {"skipped": True, "engine": "vllm", "regime": regime, "concurrency": concurrency, "repeat": repeat}
        result = run_benchmark_impl("vllm", regime, concurrency, repeat)
        result["cold_start_seconds"] = round(self.cold_start, 1)
        return result

    @modal.exit()
    def stop_server(self):
        if hasattr(self, "proc") and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=10)


@app.local_entrypoint()
def main(regime: str = "all"):
    regimes = ["short", "long"] if regime == "all" else [regime]
    engine = "vllm"

    print(f"=== inference-bench: {engine} ===")
    print(f"  Model:       {MODEL_FP16}")
    print(f"  GPU:         {GPU_TYPE}")
    print(f"  Regimes:     {regimes}")
    print(f"  Concurrency: {CONCURRENCY_LEVELS}")
    print(f"  Repeats:     {REPEATS}")

    bench = VllmBench()
    skipped = 0
    all_results = []
    for reg in regimes:
        conc_levels = CONCURRENCY_LEVELS if reg == "short" else LONG_CONCURRENCY_LEVELS
        n_repeats = REPEATS if reg == "short" else LONG_REPEATS
        for conc in conc_levels:
            for rep in range(1, n_repeats + 1):
                run_id = f"{engine}_{reg}_c{conc}_r{rep}"
                if _already_done(engine, reg, conc, rep):
                    print(f"  [skip] {run_id} already done (local)")
                    skipped += 1
                    continue
                print(f"\n  >> {engine} | {reg} | c={conc} | r={rep}/{n_repeats}")
                result = bench.run.remote(regime=reg, concurrency=conc, repeat=rep)
                if result.get("skipped"):
                    print(f"  [skip] {run_id} already done (Volume)")
                    skipped += 1
                else:
                    print_result_line(result)
                    all_results.append(result)

    if all_results:
        write_summary(all_results)
    if skipped:
        print(f"\n  Skipped {skipped} already-completed configs. Re-run reaggregate.py to merge all results.")
