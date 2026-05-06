# Inference Bench — Project Context

## Goal

Head-to-head LLM serving benchmark of vLLM, SGLang, and llama.cpp across FP16, AWQ quantization, and reasoning workloads on a single NVIDIA L4 GPU via Modal, with throughput/TTFT/TPOT/latency metrics, interactive dashboard, and portfolio entry.

## Constraints & Preferences

- **GPU**: NVIDIA L4 (24 GB) via Modal (per-second billing, no local GPU)
- **Package manager**: `uv` (not pip)
- **Architecture**: Client and server collocated in same Modal container (localhost:8000) — eliminates network variance
- **Separate Modal app files per engine** — avoids building unused Docker images
- **llama.cpp**: Binary built locally via Docker, uploaded to Modal Volume (no GPU-time compilation)
- **Budget**: ~$10-20 in Modal credits total (~$12 spent so far)
- **Decoding**: Greedy only (temperature=0)
- **Workload regimes**: short (256in/128out), long (2048in/512out), reasoning (~256in/8192out)
- **AWQ quantization**: Separate engine variants (`vllm_awq`, `sglang_awq`)
- **Reasoning**: Qwen3-8B with thinking tokens, 10 measure requests per config, conc [1,4,16], 1 repeat
- **Modal apps launched with `-d`** (detached) to survive local disconnect
- **TRT-LLM**: Backlogged (Option A: FP16 only, ~6-8 hrs dev, ~$1 GPU)

## Models

| Model | Size | Quantization | Use |
|---|---|---|---|
| Qwen2.5-7B-Instruct | ~15 GB | FP16 | Primary benchmark (vllm, sglang) |
| Qwen2.5-7B-Instruct-AWQ | ~5.2 GB | AWQ | AWQ benchmark (vllm_awq, sglang_awq) |
| Qwen2.5-7B-Instruct-GGUF | ~4.4 GB | Q4_K_M | llama.cpp benchmark (2-shard split) |
| Qwen3-8B | ~15 GB | FP16 | Reasoning benchmark (qwen3_vllm, qwen3_sglang) |
| Qwen3-8B-AWQ | ~4.5 GB | AWQ | Reasoning AWQ (qwen3_awq_vllm, qwen3_awq_sglang) |
| Qwen3-30B-A3B-GPTQ-Int4 | — | GPTQ Int4 | MoE — BLOCKED (incompatible with both engines) |

## Engines & Versions

| Engine | Version | Key Flags |
|---|---|---|
| vLLM | v0.8.5 | `--max-num-seqs 64 --gpu-mem-util 0.90` |
| SGLang | v0.4.6 | `--max-running-req 64 --mem-frac 0.85 --disable-cuda-graph` |
| llama.cpp | b5540 | `-np 4 --parallel 4 -ngl 99 -c 16384` |
| vLLM AWQ | v0.8.5 | `--quantization awq --enforce-eager` |
| SGLang AWQ | v0.4.6 | `--quantization awq --disable-cuda-graph` |
| Qwen3 vLLM | v0.8.5 | `--reasoning-parser deepseek_r1 --max-model-len 16384` |
| Qwen3 SGLang | v0.4.6 | `--reasoning-parser deepseek-r1 --disable-cuda-graph` |
| Qwen3 AWQ vLLM | v0.8.5 | `--quantization awq --reasoning-parser deepseek_r1 --max-model-len 16384` |
| Qwen3 AWQ SGLang | v0.4.6 | `--quantization awq --reasoning-parser deepseek-r1 --disable-cuda-graph` |

## Project Structure

```
inference-bench/
├── bench_lib.py               # Shared constants, image builders, benchmark logic
├── modal_vllm.py              # vLLM FP16 (standalone app)
├── modal_sglang.py            # SGLang FP16 (standalone app)
├── modal_llamacpp.py          # llama.cpp (standalone app)
├── modal_vllm_awq.py          # vLLM AWQ (standalone app)
├── modal_sglang_awq.py        # SGLang AWQ (standalone app)
├── modal_qwen3_vllm.py        # Qwen3 vLLM reasoning
├── modal_qwen3_sglang.py      # Qwen3 SGLang reasoning
├── modal_qwen3_awq_vllm.py    # Qwen3 AWQ vLLM reasoning
├── modal_qwen3_awq_sglang.py  # Qwen3 AWQ SGLang reasoning
├── modal_qwen3_moe_vllm.py    # Qwen3 MoE vLLM (non-functional)
├── modal_qwen3_moe_sglang.py  # Qwen3 MoE SGLang (non-functional)
├── modal_app.py               # Orchestrator (all FP16 engines)
├── scripts/
│   ├── generate_workload.py   # Prompt generation → prompts/workload.jsonl
│   ├── reaggregate.py         # Raw JSONL → summary.jsonl
│   ├── collect_metrics.py     # summary.jsonl → summary.csv
│   └── plot_results.py        # CSV → charts
├── prompts/
│   └── workload.jsonl          # 230 prompts (100 short + 100 long + 30 reasoning)
├── results/
│   ├── raw/                    # 102 per-request JSONL files
│   ├── summary.jsonl           # 102 aggregated run entries
│   ├── summary.csv             # 53 deduplicated rows
│   └── plots/                  # Chart PNGs
├── docs/
│   ├── index.html              # Interactive dashboard (self-contained)
│   ├── chart.umd.min.js        # Chart.js (vendored)
│   └── favicon.svg
├── docker/                     # llama.cpp Dockerfile for binary build
├── configs/                    # YAML configs (engines, workload)
├── Makefile                    # All targets
└── context.md                  # This file
```

## Benchmark Pipeline

1. **Generate prompts**: `python scripts/generate_workload.py` → `prompts/workload.jsonl` (230 prompts, seed=42)
2. **Run benchmarks**: Each `modal_*.py` starts a server, warms up, sends concurrent streaming requests, records per-request metrics to raw JSONL
3. **Aggregate**: `python scripts/reaggregate.py` → `results/summary.jsonl` (raw JSONL → per-config summary)
4. **Collect**: `python scripts/collect_metrics.py` → `results/summary.csv` (deduplicate across repeats)
5. **Dashboard**: `docs/index.html` is self-contained with inline JS data (manually updated from CSV values)

## Results Summary

### Raw Data: 102 JSONL files across 9 engines × 3 regimes

| Engine | Short | Long | Reasoning | Total Files |
|---|---|---|---|---|
| vllm | 15 (c=1,4,16,32,64 × 3r; c=1,4,16 × 1r for c=32,64) | 3 | — | 18 |
| sglang | 15 | 3 | — | 18 |
| llamacpp | 18 | 3 | — | 21 |
| vllm_awq | 15 | 3 | — | 18 |
| sglang_awq | 15 | 3 | — | 18 |
| qwen3_vllm | — | — | 3 | 3 |
| qwen3_sglang | — | — | 3 | 3 |
| qwen3_awq_vllm | — | — | 3 | 3 |
| qwen3_awq_sglang | — | — | 3 | 3 |
| **Total** | | | | **102** |

### Key Numbers

**Short regime (c=64, median across repeats)**:

| Engine | Throughput (tok/s) | TTFT p50 (ms) | Latency p95 (ms) | Success |
|---|---|---|---|---|
| vLLM AWQ | **975.6** | 651 | 9,440 | 100% |
| SGLang | 913.9 | 314 | 9,123 | 100% |
| vLLM | 831.3 | 582 | 10,222 | 100% |
| SGLang AWQ | 505.8 | 516 | 16,413 | 100% |
| llama.cpp | 189.2 | N/A | 73,360 | 100% |

**Long regime (c=64)**:

| Engine | Throughput (tok/s) | TTFT p50 (ms) | Latency p95 (ms) | Success |
|---|---|---|---|---|
| vLLM AWQ | **893.5** | 700 | 40,953 | 100% |
| SGLang | 839.6 | 889 | 40,069 | 100% |
| vLLM | 776.5 | 681 | 44,137 | 100% |
| SGLang AWQ | 486.9 | 637 | 68,502 | 100% |
| llama.cpp | 166.8 | N/A | 296,989 | 55% |

**Reasoning regime (Qwen3-8B)**:

| Engine | c=1 tok/s | c=4 tok/s | c=16 tok/s | TTFT→Answer c=4 (s) |
|---|---|---|---|---|
| Qwen3 AWQ vLLM | 25.0 | 99.2 | **345.5** | 145 |
| Qwen3 vLLM | 15.4 | 54.8 | 174.2 | 202 |
| Qwen3 SGLang | 15.8 | 55.8 | 147.0 | 298 |
| Qwen3 AWQ SGLang | 8.0 | 29.0 | 110.5 | 619 |

## Dashboard

`docs/index.html` — self-contained interactive dashboard with:
- **BLUF layout**: Winners → Findings → Methodology → Process → Results → Reproduce → Limitations
- **Regime toggle**: Short / Long / Reasoning (different engines per regime)
- **3 charts**: Throughput vs Concurrency, TTFT p50, E2E Latency p95
- **Data table**: Auto-generated per regime with reasoning-specific columns (thinking tokens, answer tokens, TTFT-to-answer)
- **5+4 engine colors**: Blue/Red/Green for FP16, Light Blue/Pink for AWQ, Indigo/Orange/Purple/Tan for Qwen3
- **Chart.js** vendored as `chart.umd.min.js`
- **Powered by Modal** chip above hero

Deployed via GitHub Pages at `ree2raz.github.io/inference-bench`.

## Key Technical Decisions

1. **`@modal.cls` with enter/method/exit** — Correct pattern for server+client in one container. Server starts in `@modal.enter()`, benchmark runs in `@modal.method()`, cleanup in `@modal.exit()`.
2. **`run_all()` single-call pattern** — Entire benchmark loop inside one `.remote()` call; survives `--detach`. Each config writes results to Volume immediately so partial progress is preserved.
3. **Skip-existing logic** — Each run checks if `raw/{engine}_{regime}_c{conc}_r{rep}.jsonl` exists in the Volume before running. Re-runs resume from where they left off.
4. **SGLang AWQ image**: Custom `make_sglang_awq_image()` builds from scratch — installs `vllm==0.7.2` for AWQ kernel operators + `flashinfer-python` from `torch2.5` channel. This downgrades PyTorch to 2.5.1, causing 3x throughput penalty (8 tok/s vs 25 tok/s for vLLM AWQ).
5. **Reasoning parser**: vLLM uses `--reasoning-parser deepseek_r1` (underscore); SGLang uses `--reasoning-parser deepseek-r1` (hyphen).
6. **SGLang context arg**: `--context-length` (not `--max-model-len` which is vLLM-only).
7. **vLLM FP16 OOM fix**: `--max-model-len 16384` caps Qwen3-8B's 40K default context. 40K needs 5.62 GiB KV cache, only 3.83 GiB available on L4 for FP16 model.
8. **Modal timeout**: Increased from 180 min to 360 min for reasoning benchmarks. c=1 reasoning at 8 tok/s for SGLang AWQ takes ~143 min alone.
9. **httpx streaming timeout**: Increased from 600s to 1800s. At c=16, FP16 vLLM internally queues requests due to KV cache pressure — some wait >600s before first token.
10. **llama.cpp non-streaming**: No TTFT/TPOT available (lacks `stream_options.include_usage`).
11. **Reaggregate regex**: `(.+?)_(short|long|reasoning)_c(\d+)_r(\d+)\.jsonl` — handles multi-underscore engine names like `qwen3_awq_sglang`.

## MoE — Blocked

Qwen3-30B-A3B-GPTQ-Int4 is incompatible with both engines on L4:

- **vLLM 0.8.5**: `FusedMoE assert self.quant_method is not None` — can't load GPTQ MoE at all
- **SGLang 0.4.6**: `NameError: name 'check_marlin_supported' is not defined` in gptq.py
- **SGLang upgrade to ≥0.5.9 failed**: cu129 Docker images too large (16.8GB → Modal build runner OOMs), `pip install --no-deps sglang==0.5.11` breaks runtime deps, `pip install sglang[all]` pulls CUDA 13.x incompatible with cu124

`make_sglang_moe_image()` and `modal_qwen3_moe_*.py` files kept for future revisit.

## Image Builders (bench_lib.py)

| Builder | Base Image | Key Installs |
|---|---|---|
| `make_vllm_image()` | `vllm/vllm-openai:v0.8.5` | httpx, numpy |
| `make_sglang_image()` | `lmsysorg/sglang:v0.4.6-cu124` | httpx, numpy |
| `make_llamacpp_image()` | `nvidia/cuda:12.4.1-runtime-ubuntu22.04` | huggingface_hub, httpx, numpy |
| `make_vllm_awq_image()` | Same as vLLM | Same as vLLM |
| `make_sglang_awq_image()` | `lmsysorg/sglang:v0.4.6-cu124` | vllm==0.7.2, flashinfer-python (torch2.5) |
| `make_sglang_moe_image()` | `lmsysorg/sglang:v0.4.6-cu124` | sglang --no-deps upgrade |

## Modal Volumes

| Volume | Purpose |
|---|---|
| `inference-bench-hf-cache` | HuggingFace model cache (persisted across runs) |
| `inference-bench-results` | Raw JSONL result files |

## Findings

1. **SGLang leads throughput at every concurrency level** — 914 tok/s vs vLLM's 831 tok/s at c=64 short (+10%)
2. **vLLM has the lowest TTFT at low concurrency** — 70 ms vs SGLang's 119 ms at c=1 (42% faster)
3. **llama.cpp fastest at c=1 but degrades beyond c=4** — 47 tok/s at c=1, plateaus at 190 tok/s at c=64
4. **Per-request throughput nearly constant for vLLM/SGLang** — ~13-17 tok/s regardless of concurrency
5. **Long regime amplifies differences** — llama.cpp 55% success at c=64 long
6. **AWQ on vLLM is a free lunch** — 2.5x throughput, 1/3 VRAM, +17% over FP16 at c=64
7. **Reasoning workloads expose engine differences** — AWQ vLLM 345 tok/s at c=16 vs FP16 SGLang 147 tok/s (2.4x)
8. **TTFT to answer is the real metric for reasoning models** — 145s (AWQ vLLM c=4) to 664s (FP16 SGLang c=16)

## Cost Breakdown

| Run | GPU Time | Estimated Cost |
|---|---|---|
| FP16 short+long (vllm, sglang, llamacpp) | ~6 hrs | ~$1.80 |
| AWQ short+long (vllm_awq, sglang_awq) | ~8 hrs | ~$2.40 |
| Qwen3 reasoning (4 engines × 3 conc) | ~14 hrs | ~$4.20 |
| Failed MoE attempts | ~4 hrs | ~$1.20 |
| Retries (timeout fixes) | ~3 hrs | ~$0.90 |
| **Total** | **~35 hrs** | **~$10.50** |

## Known Issues

- **SGLang AWQ torch 2.5.1 downgrade**: `flashinfer-python` from torch2.5 channel downgrades PyTorch, causing 3x throughput penalty. Fix: find a torch2.6-compatible flashinfer build.
- **llama.cpp non-streaming**: No TTFT/TPOT metrics available.
- **Reasoning c=16 for FP16**: KV cache can't hold 16 × 8K sequences simultaneously on L4 (only ~11 effective concurrent). vLLM internally queues.
- **SGLang FP16 reasoning c=16**: 90% success rate (1 request timed out at 600s — now mitigated with 1800s timeout).

## Git

- **Repo**: https://github.com/ree2raz/inference-bench
- **Branch**: master
- **Deployment**: GitHub Pages → docs/index.html
- **Raw results**: Gitignored (`results/raw/` in `.gitignore`), stored in Modal Volume
