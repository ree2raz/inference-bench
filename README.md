# Inference Bench — LLM Serving Benchmark Suite

**[Dashboard →](https://llm-bench.rituraj.info)** · **[Calculator →](https://llm-cost.rituraj.info)** · **[Blog post →](https://rituraj.info/posts/on-prem-llm-deployment-cto/)**

Reproducible head-to-head benchmarks of **vLLM**, **SGLang**, and **llama.cpp** across FP16, AWQ quantization, reasoning workloads, and **MoE models** on NVIDIA L4 and A100 GPUs (via Modal). 12 engine configs, 4 workload regimes, 125 benchmark runs. Measures throughput, TTFT/TPOT, tail latency, and success rate.

**Headline finding:** _SGLang leads aggregate throughput (+10% over vLLM at c=64). vLLM leads first-token latency (42% faster TTFT at c=1). AWQ quantization on vLLM is a free lunch — 2.5x throughput, 1/3 VRAM. MoE models (Qwen3-30B-A3B) achieve 3-4x faster decode than dense 7B on the same GPU by exploiting the 3.3B active parameter budget._

**Companion artifact:** This benchmark validates the throughput model in the [LLM Deploy Cost Calculator](https://llm-cost.rituraj.info) — same GPU, same model, theoretical vs measured.

## TL;DR — Pick Your Engine

| Use case | Engine | Key number |
|---|---|---|
| Interactive chatbot / copilot | **vLLM** | 42% lower TTFT at c=1 (70 ms vs 119 ms) |
| High-throughput API / batch job | **SGLang** | +10% aggregate throughput at c=64 (914 vs 831 tok/s) |
| Edge / memory-constrained | **llama.cpp** | 4.4 GB VRAM, fastest single-request E2E latency |
| Reasoning / thinking models | **vLLM AWQ** | 4.6x faster time-to-answer vs SGLang (145 s vs 664 s at c=4) |
| MoE models (Qwen3-MoE, DeepSeek) | **vLLM** | Size VRAM on total params, throughput on active params |

**AWQ on vLLM is almost always worth it** — 2.5x throughput, 1/3 VRAM, no meaningful downside if a pre-quantized checkpoint exists.

**What to trust:** Relative rankings between engines on L4 are the durable finding — they reflect architectural differences, not version-specific numbers. A May 2026 v2 sweep (vLLM v0.20.1 + SGLang :latest) confirmed L4 numbers stable within 3%; the A100 tab on the dashboard uses the v2 numbers. See [Limitations](#limitations) for full scope.

**What's not covered:** H100/B200, FP8 quantization, multi-GPU tensor parallelism, prefix caching, speculative decoding, AMD GPUs. Results are from L4 (24 GB), A100 40GB (dense/AWQ), and A100 80GB (MoE BF16) with Qwen2.5-7B (dense), Qwen3-8B (reasoning), and Qwen3-30B-A3B (MoE).

## Results at a Glance

**Short regime (c=64, Qwen2.5-7B)**:

| Engine | Throughput (tok/s) | TTFT p50 (ms) | Latency p95 (ms) |
|---|---|---|---|
| vLLM AWQ | **976** | 651 | 9,440 |
| SGLang FP16 | **914** | 314 | 9,123 |
| vLLM FP16 | 831 | 582 | 10,222 |
| SGLang AWQ | 506 | 516 | 16,413 |
| llama.cpp Q4_K_M | 189 | — | 73,360 |

**Reasoning regime (Qwen3-8B, ~6000 thinking tokens)**:

| Engine | c=1 tok/s | c=16 tok/s | TTFT→Answer c=4 (s) |
|---|---|---|---|
| Qwen3 AWQ vLLM | 25 | **346** | **145** |
| Qwen3 vLLM | 15 | 174 | 202 |
| Qwen3 SGLang | 16 | 147 | 298 |
| Qwen3 AWQ SGLang | 8 | 111 | 619 |

Interactive dashboard: [llm-bench.rituraj.info](https://llm-bench.rituraj.info)

### A100 Benchmark (vLLM v0.20.1 + SGLang :latest, Qwen2.5-7B)

May 2026 sweep on A100 40GB with current engine versions — both engines, FP16 and AWQ Marlin.

| Config | Engine | c=1 tok/s | c=16 tok/s | c=64 tok/s |
|---|---|---|---|---|
| FP16 | vLLM v0.20.1 | **80** | **1,094** | **3,102** |
| FP16 | SGLang :latest | 61 | 829 | 2,141 |
| AWQ Marlin | vLLM v0.20.1 | **177** | **2,248** | **4,762** |
| AWQ Marlin | SGLang :latest | 195 | 2,325 | 2,231 |

**Key findings from v2 sweep:** vLLM v0.20.1 FP16 is +11% faster than v0.8.5 at c=1 (80 vs 72 tok/s) and +21% at c=64. Marlin kernels improved +70% in v0.20.1 (177 vs 104 tok/s). On A100, vLLM leads SGLang at high concurrency — SGLang Marlin collapses to 2,231 tok/s at c=64 vs vLLM's 4,762 tok/s, the opposite of the L4 pattern where SGLang wins at c=64.

### MoE Benchmark (vLLM v0.20.1, Qwen3-30B-A3B)

Tests the decode efficiency of mixture-of-experts models. Qwen3-30B-A3B has 30.5B total parameters but only 3.3B active per token — the hypothesis is that MoE throughput should approach that of a dense 3.3B model, not a 30.5B model.

| Config | GPU | c=1 tok/s | c=4 tok/s | c=16 tok/s |
|---|---|---|---|---|
| BF16 | A100 80GB | 134 | 301 | —† |
| AWQ Marlin | A100 40GB | 165 | 520 | 1,476 |
| AWQ Marlin 16K ctx | A100 40GB | 161 | 508 | 1,485 |

_† BF16 c=16 not feasible — 61 GB weights exhausts 80 GB VRAM with KV cache._

**Key finding:** MoE decode throughput is ~2-3x faster than dense 7B on the same GPU class. At c=1 on A100, MoE BF16 delivers 134 tok/s vs dense 7B's 92 tok/s (1.5x, vLLM v0.20.2). MoE AWQ at c=16 hits 1,476 tok/s vs dense AWQ's 811 tok/s on A100 40GB (1.8x). However, the efficiency gap is significant — MoE BF16 achieves 46% efficiency vs dense FP16's 63% on A100 (both relative to their active-param theoretical bandwidth), with expert routing overhead consuming the remaining bandwidth. AWQ Marlin MoE efficiency drops to 11-18% due to the combined overhead of dequantization and expert loading. **Long context (16K) has negligible impact** on short-prompt throughput — the KV cache growth doesn't affect decode speed until sequences actually approach the limit.

## Validated by Real Benchmarks

The throughput model in the [LLM Deploy Cost Calculator](https://llm-cost.rituraj.info) predicts decode performance from GPU specs. These benchmarks measure the actual numbers on the same hardware and model:

| Config | Theoretical | Measured | Efficiency |
|---|---|---|---|
| FP16 c=1 per-stream | 21.4 tok/s | 17.1 tok/s | 80% |
| AWQ c=1 per-stream | 85.4 tok/s | 43.2 tok/s | 51% |
| FP16 c=64 aggregate | 1,294 tok/s | 914 tok/s (SGLang) | 71% |
| FP16 c=64 aggregate | 1,294 tok/s | 831 tok/s (vLLM) | 64% |
| FP16 c=1 per-stream (A100 40GB, v0.8.5) | 111 tok/s | 72.1 tok/s | 65% |
| AWQ Marlin c=1 (A100 40GB, v0.8.5) | 444 tok/s | 104.2 tok/s | 23% |
| FP16 c=64 per-stream (A100 40GB, v0.8.5) | 111 tok/s | 40.1 tok/s | 36% |
| FP16 c=1 spot-check (A100 80GB, v0.20.2) | 145.6 tok/s | 92.3 tok/s | 63% |

FP16 achieves 64-80% of theoretical bandwidth — the gap comes from kernel overhead, attention computation, and KV cache reads. AWQ drops to 51% due to dequantization overhead and irregular memory access patterns. The calculator's "ideal batching" assumption is real: engines achieve 64-71% of ideal at high concurrency.

A100 40GB achieves lower per-stream efficiency (36-65% vs 61-80% on L4) because at 1.55 TB/s memory bandwidth, the bottleneck shifts from memory to compute (attention, KV cache). The higher aggregate throughput (2,564 tok/s at c=64 vs L4's 831) comes from the A100's 10x compute advantage (312 vs 31 TFLOPS), not bandwidth. AWQ Marlin on A100 is 48-64% faster than default AWQ, confirming Marlin kernels as essential for quantized inference on Ampere+ GPUs. A v0.20.2 spot-check on A100 80GB showed 92.3 tok/s at c=1 — a ~28% throughput improvement over v0.8.5 on the 40GB variant.

## Setup

### Hardware

| Component | L4 | A100 |
|---|---|---|
| GPU | NVIDIA L4 (24 GB VRAM) | NVIDIA A100 40GB (dense/AWQ) · A100 80GB (MoE BF16) |
| Provider | Modal (cloud GPU, per-second billing) | Modal (cloud GPU, per-second billing) |
| CUDA | 12.4.1 | 12.4.1 |

### Models

| Model | Size | Quantization | Use |
|---|---|---|---|
| Qwen2.5-7B-Instruct | ~15 GB | FP16 | Primary benchmark (vllm, sglang) |
| Qwen2.5-7B-Instruct-AWQ | ~5.2 GB | AWQ 4-bit | Quantization benchmark (vllm_awq, sglang_awq) |
| Qwen2.5-7B-Instruct-GGUF | ~4.4 GB | Q4_K_M | llama.cpp benchmark |
| Qwen3-8B | ~15 GB | FP16 | Reasoning benchmark (qwen3_vllm, qwen3_sglang) |
| Qwen3-8B-AWQ | ~4.5 GB | AWQ 4-bit | Reasoning + quantization (qwen3_awq_vllm, qwen3_awq_sglang) |

### Engine Versions

| Engine | Version | GPU | Key Flags |
|---|---|---|---|
| vLLM | v0.8.5 | L4 | `--max-num-seqs 64 --gpu-mem-util 0.90` |
| SGLang | v0.4.6 | L4 | `--max-running-req 64 --mem-frac 0.85 --disable-cuda-graph` |
| llama.cpp | b5540 | L4 | `-np 4 --parallel 4 -ngl 99 -c 16384` |
| vLLM AWQ | v0.8.5 | L4 | `--quantization awq --enforce-eager` |
| SGLang AWQ | v0.4.6 | L4 | `--quantization awq --disable-cuda-graph` |
| Qwen3 vLLM | v0.8.5 | L4 | `--reasoning-parser deepseek_r1 --max-model-len 16384` |
| Qwen3 SGLang | v0.4.6 | L4 | `--reasoning-parser deepseek-r1 --disable-cuda-graph` |
| Qwen3 AWQ vLLM | v0.8.5 | L4 | `--quantization awq --reasoning-parser deepseek_r1 --max-model-len 16384` |
| Qwen3 AWQ SGLang | v0.4.6 | L4 | `--quantization awq --reasoning-parser deepseek-r1 --disable-cuda-graph` |
| MoE BF16 vLLM | v0.20.1 | A100 80GB | `vllm serve Qwen/Qwen3-30B-A3B --no-enable-log-requests` |
| MoE AWQ vLLM | v0.20.1 | A100 40GB | `vllm serve ... --quantization awq_marlin --no-enable-log-requests` |
| vLLM (A100 v2) | v0.20.1 | A100 40GB | `vllm serve ... --max-num-seqs 64 --no-enable-log-requests` |
| SGLang (A100 v2) | :latest | A100 40GB | `--max-running-req 64 --mem-frac 0.85 --disable-cuda-graph` |

### Workload Regimes

- **Short:** ≤256 input tokens, 128 max output (chat-style). Concurrency: 1, 4, 16, 32, 64. 3 repeats per config, median reported.
- **Long:** ≤2048 input tokens, 512 max output (RAG-style). Concurrency: 1, 16, 64. 1 repeat per config.
- **Reasoning:** ≤256 input tokens, ~6000 thinking + answer tokens (Qwen3 with `enable_thinking`). Concurrency: 1, 4, 16. 1 repeat, 10 measure requests per config.
- **Sampling:** Greedy (temperature=0), deterministic (seed=42)
- **Client-server colocation:** Same Modal container, localhost:8000. Eliminates network variance but understates real-world TTFT by ~5-20ms.

## Benchmark Pipeline

### 1. Generate prompts

```bash
python scripts/generate_workload.py
# → prompts/workload.jsonl (230 prompts: 100 short + 100 long + 30 reasoning)
```

Deterministic generation (seed=42). Two template families:
- **Short:** 50 CS/systems topics as single-turn questions (~200 input tokens, 128 max output)
- **Long:** 10 domain pairs × 10 analytical questions with synthetic RAG context (~1800 input tokens, 512 max output)
- **Reasoning:** 30 multi-step reasoning prompts for Qwen3 thinking models

### 2. Run benchmarks

Each `modal_*.py` starts a server in `@modal.enter()`, warms up with 2×concurrency requests, then runs the benchmark in `@modal.method()`, cleaning up in `@modal.exit()`. Results write to Modal Volume immediately — interrupted runs resume from where they left off (skip-existing logic).

```bash
# Requires: Modal account, ~$10 in credits
git clone https://github.com/ree2raz/inference-bench
cd inference-bench
make setup          # venv + deps + modal auth

# Upload llama.cpp binary to Modal Volume (one-time, ~15 min local build)
bash scripts/build_llamacpp_local.sh
modal volume put inference-bench-hf-cache /tmp/opencode/llamacpp-build/llama-server /llamacpp/
# ... (5 more shared libs)

# FP16 engines (~3-4 hours, ~$4)
modal run modal_vllm.py --regime short     # ~30 min
modal run modal_vllm.py --regime long       # ~5 min
modal run modal_sglang.py --regime short    # ~30 min
modal run modal_sglang.py --regime long     # ~55 min
modal run modal_llamacpp.py                 # ~2-3 hours

# AWQ quantization variants (~2 hours, ~$2.40)
modal run modal_vllm_awq.py --regime short
modal run modal_vllm_awq.py --regime long
modal run modal_sglang_awq.py --regime short
modal run modal_sglang_awq.py --regime long

# Qwen3 reasoning benchmarks (~14 hours, ~$4.20)
modal run modal_qwen3_vllm.py -d
modal run modal_qwen3_sglang.py -d
modal run modal_qwen3_awq_vllm.py -d
modal run modal_qwen3_awq_sglang.py -d

# Generate metrics + charts
make report
```

All Modal apps launched with `-d` (detached) to survive local disconnects.

### 3. Aggregate & collect

```bash
python scripts/reaggregate.py    # raw JSONL → summary.jsonl
python scripts/collect_metrics.py # summary.jsonl → summary.csv
```

### 4. Customize

**Change the model** — edit `configs/engines.yaml`:
```yaml
# vLLM/SGLang use HF repo IDs
model: "meta-llama/Meta-Llama-3-8B-Instruct"

# llama.cpp — point to a GGUF repo + file
engines:
  llamacpp:
    gguf_model: "bartowski/Meta-Llama-3-8B-Instruct-GGUF"
    gguf_file: "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
```

Update server flags in `bench_lib.py` (`VLLM_SERVER_ARGS`, `SGLANG_SERVER_ARGS`, `start_llamacpp_server`).

**Change the workload** — edit `configs/workload_short.yaml` and `workload_long.yaml`:
```yaml
max_input_tokens: 256
max_output_tokens: 128
concurrency_levels: [1, 4, 16, 32, 64]
repeats: 3
```

**Change the GPU** — edit `GPU_TYPE` in `bench_lib.py`. Modal supports A10G, A100, H100, L4, T4, and more. Budget estimate: L4 ~$0.30/hr, A100 ~$1.50/hr, H100 ~$4.00/hr.

**Run a subset** — skip engines or regimes:
```bash
# Single engine only
modal run modal_vllm.py --regime short

# Combined orchestrator with filters
modal run modal_app.py --engine sglang --regime long

# Parallel all FP16 engines (fastest, if you have Modal credits)
make bench-all-parallel
```

Already-completed configs are automatically skipped — re-running resumes from where you left off.

## Findings

1. **SGLang leads throughput; vLLM leads first-token latency.** At c=64 short, SGLang hits 914 tok/s vs vLLM's 831 tok/s (+10%). But at c=1, vLLM's TTFT is 70ms vs SGLang's 119ms (42% faster). Both maintain ~13-17 tok/s per-request regardless of concurrency.

2. **AWQ quantization on vLLM is a free lunch.** 2.5x throughput at c=1 (43 tok/s vs 17), 1/3 VRAM (5.2 GB vs 15 GB). At c=64, AWQ vLLM hits 976 tok/s. For reasoning workloads, AWQ vLLM reaches 345 tok/s at c=16 — 2.4x faster than FP16 SGLang (147 tok/s). Exception: SGLang AWQ suffers a torch 2.5.1 compatibility issue, degrading to 8 tok/s at c=1.

3. **llama.cpp wins at c=1, loses at scale.** 47 tok/s at c=1 with 2.7s p95 latency (2.8x faster E2E than FP16). But limited parallelism (`--parallel 4`) caps throughput at 190 tok/s by c=64, and long-regime success rate drops to 55%. Best for edge, embedded, and low-concurrency use cases.

4. **Reasoning workloads change the metric.** With ~6000 thinking tokens per request, first-answer token takes 145-664 seconds. AWQ vLLM at c=4 delivers answers 4.6x faster than SGLang at c=16. For thinking models, "time to useful output" is what matters, not TTFT.

5. **Long regime amplifies every difference.** At c=64 long, SGLang is 8% faster than vLLM with 9% lower p95 latency. llama.cpp drops to 55% success. Longer sequences stress KV cache management and batching efficiency.

6. **MoE models punch above their weight class.** Qwen3-30B-A3B (3.3B active) on A100 achieves 134 tok/s BF16 at c=1 and 1,476 tok/s AWQ at c=16 — 1.5x and 1.8x faster than dense 7B on the same GPU (dense baseline updated to vLLM v0.20.2). The active-params model holds: MoE decode throughput scales with active parameters, not total parameters. But expert routing overhead is real — MoE BF16 achieves only 46% of theoretical bandwidth vs dense FP16's 63% (vLLM v0.20.2), and AWQ Marlin MoE drops to 11-18% due to combined dequantization + expert loading overhead. Long context (16K) has no measurable impact on short-prompt throughput.

## What This Doesn't Measure

- Multi-GPU / tensor parallelism
- Speculative decoding
- LoRA / adapter hot-swapping
- Long-context behavior beyond 4K tokens (short regime)
- Mixed workloads (some long, some short concurrent)
- Cold-start under autoscaling (partial — we measure first-token cold start)
- CPU-only / Metal / Apple Silicon performance (llama.cpp's natural habitat)
- Structured output / JSON mode
- Tool use / function calling overhead

## Limitations

- **Two GPU classes tested** (L4, A100). Results may differ on H100/B200.
- **Fixed model set** (Qwen2.5-7B, Qwen3-8B, Qwen3-30B-A3B MoE).
- **Greedy decoding only.** Sampling with temperature > 0 may change throughput characteristics.
- **Synthetic prompts.** Not drawn from a real production workload.
- **N=3 short, N=1 long** per config. Sufficient for median trends but not for rigorous statistical claims.
- **Modal network overhead.** ~5-20ms internal network latency included in TTFT and latency numbers (not subtracted).
- **No quantization parity for llama.cpp.** Q4_K_M vs FP16/BF16 — cross-engine absolute comparison is fair, but llama.cpp reflects its quantized model.
- **llama.cpp non-streaming.** TTFT and TPOT not available.
- **SGLang AWQ torch 2.5.1 downgrade.** flashinfer-python forces PyTorch downgrade, causing ~3x throughput penalty. Not representative of SGLang AWQ on a compatible torch version.

## Cost Breakdown

| Run | GPU Time | Estimated Cost |
|---|---|---|
| FP16 short+long (3 engines) | ~6 hrs | ~$1.80 |
| AWQ short+long (2 engines) | ~8 hrs | ~$2.40 |
| Qwen3 reasoning (4 engines) | ~14 hrs | ~$4.20 |
| Failed MoE attempts (early vLLM CLI) | ~4 hrs | ~$1.20 |
| MoE benchmarks (BF16+AWQ+long) | ~2 hrs | ~$3.50 |
| Retries (timeout fixes) | ~3 hrs | ~$0.90 |
| A100 validation (FP16 + AWQ + Marlin) | ~0.5 hrs | ~$0.75 |
| v2 sweep (vLLM v0.20.1 + SGLang :latest, L4 + A100) | ~4 hrs | ~$5.50 |
| **Total** | **~41.5 hrs** | **~$20.25** |

## Project Structure

```
inference-bench/
├── bench_lib.py               # shared constants, image builders, benchmark logic
├── modal_vllm.py              # vLLM FP16 (standalone)
├── modal_sglang.py            # SGLang FP16 (standalone)
├── modal_llamacpp.py          # llama.cpp (standalone)
├── modal_vllm_awq.py          # vLLM AWQ (standalone)
├── modal_sglang_awq.py        # SGLang AWQ (standalone)
├── modal_qwen3_vllm.py        # Qwen3 vLLM reasoning
├── modal_qwen3_sglang.py      # Qwen3 SGLang reasoning
├── modal_qwen3_awq_vllm.py    # Qwen3 AWQ vLLM reasoning
├── modal_qwen3_awq_sglang.py  # Qwen3 AWQ SGLang reasoning
├── modal_app.py               # orchestrator (all FP16 engines)
├── modal_qwen3_moe_bf16.py    # MoE Qwen3-30B-A3B BF16 (A100 80GB)
├── modal_qwen3_moe_awq.py     # MoE Qwen3-30B-A3B AWQ (A100 40GB)
├── modal_qwen3_moe_awq_long.py # MoE Qwen3-30B-A3B AWQ 16K ctx (A100 40GB)
├── scripts/
│   ├── build_llamacpp_local.sh    # local Docker build for llama.cpp binary
│   ├── generate_workload.py       # generates prompts/workload.jsonl
│   ├── reaggregate.py             # raw JSONL → summary.jsonl
│   ├── collect_metrics.py         # summary.jsonl → summary.csv
│   └── plot_results.py            # CSV → charts
├── prompts/
│   └── workload.jsonl             # 230 prompts (committed, seed=42)
├── modal_v2_*.py              # v2 sweep scripts (vLLM v0.20.1 + SGLang :latest, L4 + A100)
├── results/
│   ├── raw/                       # per-request JSONL files (baseline)
│   ├── raw_v2/                    # v2 sweep JSONL files (May 2026)
│   ├── summary.jsonl              # aggregated run entries
│   ├── summary.csv                # deduplicated rows
│   └── plots/                     # chart SVGs/PNGs
├── docs/
│   ├── index.html                 # interactive dashboard (self-contained)
│   ├── chart.umd.min.js           # Chart.js v4.5.1 (vendored)
│   └── favicon.svg
├── docker/                        # llama.cpp Dockerfile for binary build
├── configs/                        # YAML configs (engines, workload)
├── Makefile
└── context.md                     # extended project notes
```

## See Also

- **[LLM Deploy Cost Calculator](https://llm-cost.rituraj.info)** — GPU sizing, cost comparison, and break-even analysis for LLM deployment. The throughput model in this calculator is validated by the benchmarks above.
- **[Blog: Four numbers that change your LLM self-hosting cost estimate](https://www.rituraj.info/posts/on-prem-llm-deployment-cto/)** — Total params for MoE VRAM, KV cache dtype, throughput bottleneck, and replica count — the four things most cost estimates get wrong.

## License

Apache 2.0