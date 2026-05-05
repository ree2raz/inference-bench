---
name: modal-inference-bench
description: Modal AI infra for inference-bench project (vLLM/SGLang/llama.cpp on Modal L4). Fetches live Modal docs on demand.
argument-hint: []
---

# Modal Inference Bench

Project: `/home/rituraj/projects/inference-bench/`

**Rule: When uncertain about any Modal API call, decorator, or CLI flag, fetch the relevant doc page FIRST. Do not guess. OpenCode hallucinates on this stack.**

---

## Project Codebase

### Files

| File | Purpose |
|------|---------|
| `bench_lib.py` | Shared helpers, image builders, server args, benchmark logic. **No `modal.App` here.** |
| `modal_app.py` | Runs all 3 engines sequentially. Orchestrates `modal run --engine vllm\|sglang\|llamacpp\|all`. |
| `modal_vllm.py` | vLLM-only benchmark. Own `modal.App("inference-bench-vllm")`. |
| `modal_sglang.py` | SGLang-only benchmark. Own `modal.App("inference-bench-sglang")`. |
| `modal_llamacpp.py` | llama.cpp-only benchmark. Own `modal.App("inference-bench-llamacpp")`. |
| `configs/engines.yaml` | Engine versions and server flags (vLLM 0.8.5, SGLang 0.4.6, llama.cpp b5540). |
| `scripts/generate_workload.py` | Generates `prompts/workload.jsonl`. |

### Engines

- **vLLM**: `vllm/vllm-openai:v0.8.5-cu124`, Python 3.12, FP16, Qwen2.5-7B-Instruct
- **SGLang**: `lmsysorg/sglang:v0.4.6-cu124`, Python 3.11, FP16, Qwen2.5-7B-Instruct
- **llama.cpp**: `dataelement/llama.cpp:server-cuda`, Python 3.11, Q4_K_M GGUF

### Volume Names (create_if_missing=True)

- `inference-bench-hf-cache` — HuggingFace model weights
- `inference-bench-results` — benchmark results JSONL

---

## Patterns That Hallucinate Easily

These are safe to use verbatim. Anything else → fetch docs.

### App + Cls + Enter/Exit (benchmark pattern)

```python
import modal
app = modal.App("app-name")

@app.cls(
    image=make_image(),
    gpu="L4",
    timeout=60 * 45,
    scaledown_window=60,
    volumes={"/hf_cache": hf_cache, "/results": results_vol},
)
class MyBench:
    @modal.enter()   # runs once at container cold start
    def start(self):
        self.proc = subprocess.Popen(SERVER_ARGS)

    @modal.method()  # callable via .remote()
    def run(self, arg: str) -> dict:
        return {"result": "ok"}

    @modal.exit()    # cleanup on container recycle
    def stop(self):
        self.proc.terminate()
        self.proc.wait(timeout=10)
```

### Image (lazy — builds on first call)

```python
def make_vllm_image():
    return (
        modal.Image.from_registry("vllm/vllm-openai:v0.8.5", add_python="3.12")
        .pip_install("huggingface_hub", "httpx>=0.27", "numpy>=1.26")
        .add_local_file("local/path", remote_path="/remote/path")
        .env({"HF_HOME": "/hf_cache"})
    )
```

### Local Entrypoint + Remote Call

```python
@app.local_entrypoint()
def main(engine: str = "all", regime: str = "all"):
    result = MyBench().run.remote(regime=regime, concurrency=4, repeat=1)
```

```bash
modal run modal_vllm.py --regime short
modal run --build modal_app.py  # --build forces image build before run
```

### Volume

```python
hf_cache = modal.Volume.from_name("inference-bench-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("inference-bench-results", create_if_missing=True)
# Inside container: volume.commit() after writes
```

### Web Endpoint

```python
@app.function(image=make_image())
@modal.fastapi_endpoint()
def predict(item: dict):
    return {"result": item["x"] ** 2}
# Deploy: modal deploy  |  Dev: modal serve  (ephemeral URL)
```

---

## Live Doc Lookup

Fetch with `web_extract` when working with anything beyond the patterns above:

| Topic | URL |
|-------|-----|
| Python API (App, Cls, Function, decorators) | https://modal.com/docs/guide/python-api |
| Web endpoints (FastAPI, ASGI, WSGI) | https://modal.com/docs/guide/webhooks |
| Volumes & persistence | https://modal.com/docs/guide/volume |
| Images & container building | https://modal.com/docs/guide/image |
| Secrets (API keys, env vars) | https://modal.com/docs/guide/secrets |
| Scheduling (cron, periodic) | https://modal.com/docs/guide/scheduling |
| Sandboxes (ephemeral VMs) | https://modal.com/docs/guide/sandbox |
| Deploy & app lifecycle | https://modal.com/docs/guide/deploy |
| GPU configuration | https://modal.com/docs/guide/gpu |
| CLI reference | https://modal.com/docs/cli |

---

## CLI Quick Ref

```bash
modal run file.py                              # run, container starts on first .remote() call
modal run --build file.py                      # build image first, then run
modal run file.py --engine vllm --regime short  # pass args to local_entrypoint
modal serve file.py                            # dev server, ephemeral URL, live reload
modal deploy file.py                           # persistent deploy
modal app list                                 # active apps
modal app logs <name> --follow                  # tail logs
modal app stop <name>                           # stop deployed app
modal shell file.py::ClsName.method             # debug shell into running container
```

---

## How to Use This Skill with OpenCode

When OpenCode starts working on this project:

1. Read `bench_lib.py` and the relevant engine file first.
2. For any Modal API usage not in the "Patterns" section above, fetch the relevant doc URL from the lookup table.
3. Check `configs/engines.yaml` for actual version pins and server flags — do not invent flags.
4. When adding new server args or changing engine versions, update `bench_lib.py` and `configs/engines.yaml` in the same commit.
5. Volume names are defined in `bench_lib.py` — do not rename without creating the new volume first.
