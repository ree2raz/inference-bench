SHELL := /bin/bash
VENV := .venv
PYTHON := $(VENV)/bin/python
MODAL := $(VENV)/bin/modal

.PHONY: setup bench-vllm bench-sglang bench-llamacpp bench-all-parallel report clean collect plots generate-prompts download-results

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

bench-all: $(VENV)
	$(MODAL) run modal_app.py

bench-all-parallel: $(VENV)
	@echo "Launching all 3 engines in parallel..."
	$(MODAL) run modal_vllm.py --regime long & \
	$(MODAL) run modal_sglang.py --regime long & \
	$(MODAL) run modal_llamacpp.py & \
	wait
	@echo "All engines complete."

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
