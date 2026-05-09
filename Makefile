SHELL := /bin/bash
VENV := .venv
PYTHON := $(VENV)/bin/python
MODAL := $(VENV)/bin/modal

.PHONY: setup bench-vllm bench-sglang bench-llamacpp bench-vllm-awq bench-sglang-awq bench-qwen3 bench-reasoning bench-all-parallel report clean collect plots generate-prompts download-results bench-a100-latest

setup: $(VENV)
	uv pip install modal httpx numpy matplotlib pandas pyyaml
	$(MODAL) setup 2>/dev/null || true
	@echo "Setup complete. Run 'make bench-vllm' to start."

$(VENV):
	uv venv $(VENV)

generate-prompts: $(VENV)
	$(PYTHON) scripts/generate_workload.py

bench-vllm: $(VENV)
	$(MODAL) run modal_vllm.py

bench-sglang: $(VENV)
	$(MODAL) run modal_sglang.py

bench-llamacpp: $(VENV)
	$(MODAL) run modal_llamacpp.py

bench-vllm-awq: $(VENV)
	$(MODAL) run modal_vllm_awq.py

bench-sglang-awq: $(VENV)
	$(MODAL) run modal_sglang_awq.py

bench-awq: $(VENV)
	@echo "Launching AWQ benchmarks in parallel..."
	$(MODAL) run modal_vllm_awq.py & \
	$(MODAL) run modal_sglang_awq.py & \
	wait
	@echo "AWQ benchmarks complete."

bench-qwen3-vllm: $(VENV)
	$(MODAL) run modal_qwen3_vllm.py

bench-qwen3-sglang: $(VENV)
	$(MODAL) run modal_qwen3_sglang.py

bench-qwen3-awq-vllm: $(VENV)
	$(MODAL) run modal_qwen3_awq_vllm.py

bench-qwen3-awq-sglang: $(VENV)
	$(MODAL) run modal_qwen3_awq_sglang.py

bench-qwen3-moe-vllm: $(VENV)
	$(MODAL) run modal_qwen3_moe_vllm.py

bench-qwen3-moe-sglang: $(VENV)
	$(MODAL) run modal_qwen3_moe_sglang.py

bench-reasoning: $(VENV)
	@echo "Launching all 6 Qwen3 reasoning benchmarks in parallel..."
	$(MODAL) run -d modal_qwen3_vllm.py & \
	$(MODAL) run -d modal_qwen3_sglang.py & \
	$(MODAL) run -d modal_qwen3_awq_vllm.py & \
	$(MODAL) run -d modal_qwen3_awq_sglang.py & \
	$(MODAL) run -d modal_qwen3_moe_vllm.py & \
	$(MODAL) run -d modal_qwen3_moe_sglang.py & \
	wait
	@echo "All reasoning benchmarks launched."

bench-all: $(VENV)
	$(MODAL) run modal_app.py

bench-all-parallel: $(VENV)
	@echo "Launching all 3 engines in parallel..."
	$(MODAL) run modal_vllm.py --regime long & \
	$(MODAL) run modal_sglang.py --regime long & \
	$(MODAL) run modal_llamacpp.py & \
	wait
	@echo "All engines complete."

bench-a100-latest: $(VENV)
	$(MODAL) run modal_vllm_latest_check.py

download-results: $(VENV)
	mkdir -p results/raw
	$(MODAL) volume get inference-bench-results /raw/ results/raw/

collect: $(VENV)
	$(PYTHON) scripts/collect_metrics.py

plots: $(VENV)
	$(PYTHON) scripts/plot_results.py

report: collect plots
	@echo "Report generated. Check results/summary.csv and results/plots/"

clean:
	rm -rf results/raw/*.jsonl results/summary.csv results/summary.jsonl results/plots/*.png results/plots/*.svg

clobber: clean
	rm -rf $(VENV)
