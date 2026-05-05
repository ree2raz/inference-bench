"""Run all three engines sequentially. Builds all images.

Usage: modal run modal_app.py [--engine vllm|sglang|llamacpp|all] [--regime short|long|all]
"""
import modal
import subprocess
import time

from bench_lib import (
    GPU_TYPE, MODEL_FP16, CONCURRENCY_LEVELS, REPEATS,
    hf_cache, results_vol,
    make_vllm_image, make_sglang_image, make_llamacpp_image,
    wait_for_server,
    VLLM_SERVER_ARGS, SGLANG_SERVER_ARGS, start_llamacpp_server,
    run_benchmark_impl, print_result_line, write_summary,
)

app = modal.App("inference-bench")


@app.cls(
    image=make_vllm_image(),
    gpu=GPU_TYPE,
    timeout=60 * 45,
    scaledown_window=60,
    volumes={"/hf_cache": hf_cache, "/results": results_vol},
)
class VllmBench:
    @modal.enter()
    def start_server(self):
        self.proc = subprocess.Popen(VLLM_SERVER_ARGS)
        t0 = time.perf_counter()
        assert wait_for_server(timeout=300), "vLLM server failed to start"
        self.cold_start = time.perf_counter() - t0

    @modal.method()
    def run(self, regime: str, concurrency: int, repeat: int) -> dict:
        result = run_benchmark_impl("vllm", regime, concurrency, repeat)
        result["cold_start_seconds"] = round(self.cold_start, 1)
        return result

    @modal.exit()
    def stop_server(self):
        if hasattr(self, "proc") and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=10)


@app.cls(
    image=make_sglang_image(),
    gpu=GPU_TYPE,
    timeout=60 * 45,
    scaledown_window=60,
    volumes={"/hf_cache": hf_cache, "/results": results_vol},
)
class SglangBench:
    @modal.enter()
    def start_server(self):
        self.proc = subprocess.Popen(SGLANG_SERVER_ARGS)
        t0 = time.perf_counter()
        assert wait_for_server(timeout=300), "SGLang server failed to start"
        self.cold_start = time.perf_counter() - t0

    @modal.method()
    def run(self, regime: str, concurrency: int, repeat: int) -> dict:
        result = run_benchmark_impl("sglang", regime, concurrency, repeat)
        result["cold_start_seconds"] = round(self.cold_start, 1)
        return result

    @modal.exit()
    def stop_server(self):
        if hasattr(self, "proc") and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=10)


@app.cls(
    image=make_llamacpp_image(),
    gpu=GPU_TYPE,
    timeout=60 * 45,
    scaledown_window=60,
    volumes={"/hf_cache": hf_cache, "/results": results_vol},
)
class LlamacppBench:
    @modal.enter()
    def start_server(self):
        self.proc = start_llamacpp_server()
        t0 = time.perf_counter()
        assert wait_for_server(timeout=300), "llama.cpp server failed to start"
        self.cold_start = time.perf_counter() - t0

    @modal.method()
    def run(self, regime: str, concurrency: int, repeat: int) -> dict:
        result = run_benchmark_impl("llamacpp", regime, concurrency, repeat)
        result["cold_start_seconds"] = round(self.cold_start, 1)
        return result

    @modal.exit()
    def stop_server(self):
        if hasattr(self, "proc") and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=10)


engine_classes = {
    "vllm": VllmBench,
    "sglang": SglangBench,
    "llamacpp": LlamacppBench,
}


@app.local_entrypoint()
def main(engine: str = "all", regime: str = "all"):
    engines = ["vllm", "sglang", "llamacpp"] if engine == "all" else [engine]
    regimes = ["short", "long"] if regime == "all" else [regime]

    print("=" * 60)
    print("  inference-bench")
    print("=" * 60)
    print(f"  Engines:     {engines}")
    print(f"  Regimes:     {regimes}")
    print(f"  Concurrency: {CONCURRENCY_LEVELS}")
    print(f"  Repeats:     {REPEATS}")
    print(f"  GPU:         {GPU_TYPE}")
    print(f"  Model:       {MODEL_FP16}")
    print("=" * 60)

    all_results = []

    for eng in engines:
        print(f"\n{'─' * 50}")
        print(f"  Engine: {eng}")
        print(f"{'─' * 50}")

        cls = engine_classes[eng]
        bench = cls()

        for reg in regimes:
            for conc in CONCURRENCY_LEVELS:
                for rep in range(1, REPEATS + 1):
                    print(f"\n  >> {eng} | {reg} | c={conc} | r={rep}/{REPEATS}")
                    result = bench.run.remote(regime=reg, concurrency=conc, repeat=rep)
                    print_result_line(result)
                    all_results.append(result)

    write_summary(all_results)

    print(f"\n{'=' * 60}")
    print(f"  Done. {len(all_results)} configs -> results/summary.jsonl")
    print(f"  Next: make report")
    print(f"{'=' * 60}")
