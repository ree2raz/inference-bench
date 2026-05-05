"""Generate comparison plots from summary.csv."""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ENGINES = ["vllm", "vllm_awq", "sglang", "sglang_awq", "llamacpp", "qwen3_vllm", "qwen3_sglang", "qwen3_awq_vllm", "qwen3_awq_sglang", "qwen3_moe_vllm", "qwen3_moe_sglang"]
COLORS = {"vllm": "#2563eb", "vllm_awq": "#93c5fd", "sglang": "#dc2626", "sglang_awq": "#fca5a5", "llamacpp": "#16a34a", "qwen3_vllm": "#7c3aed", "qwen3_sglang": "#a855f7", "qwen3_awq_vllm": "#c084fc", "qwen3_awq_sglang": "#d8b4fe", "qwen3_moe_vllm": "#f59e0b", "qwen3_moe_sglang": "#fbbf24"}
MARKERS = {"vllm": "o", "vllm_awq": "D", "sglang": "s", "sglang_awq": "p", "llamacpp": "^", "qwen3_vllm": "v", "qwen3_sglang": "P", "qwen3_awq_vllm": "X", "qwen3_awq_sglang": "h", "qwen3_moe_vllm": "*", "qwen3_moe_sglang": "d"}


def load_csv(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def to_float(val):
    if val is None or val == "" or val == "None":
        return None
    return float(val)


def plot_throughput_vs_concurrency(rows: list[dict], output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for idx, regime in enumerate(["short", "long"]):
        ax = axes[idx]
        for engine in ENGINES:
            data = [(int(r["concurrency"]), to_float(r["throughput_median"]), to_float(r["throughput_spread"]))
                    for r in rows if r["engine"] == engine and r["regime"] == regime
                    and to_float(r["throughput_median"]) is not None]
            if not data:
                continue
            data.sort()
            concs = [d[0] for d in data]
            throughputs = [d[1] for d in data]
            spreads = [d[2] if d[2] else 0 for d in data]
            lower = [max(0, t - s) for t, s in zip(throughputs, spreads)]
            upper = [t + s for t, s in zip(throughputs, spreads)]
            ax.plot(concs, throughputs, marker=MARKERS[engine], color=COLORS[engine],
                    label=engine, linewidth=2, markersize=7)
            ax.fill_between(concs, lower, upper, color=COLORS[engine], alpha=0.15)
        ax.set_xlabel("Concurrent Requests")
        ax.set_ylabel("Throughput (output tokens/sec)")
        ax.set_title(f"Throughput vs Concurrency ({regime} regime)")
        ax.legend()
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 16, 32, 64])
        ax.set_xticklabels(["1", "4", "16", "32", "64"])
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "throughput_vs_concurrency.png")
    plt.savefig(path, dpi=150)
    plt.savefig(path.replace(".png", ".svg"))
    plt.close()
    print(f"  → {path}")


def plot_ttft_vs_concurrency(rows: list[dict], output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for idx, regime in enumerate(["short", "long"]):
        ax = axes[idx]
        for engine in ENGINES:
            data = [(int(r["concurrency"]), to_float(r["ttft_p50_median"]), to_float(r["ttft_p95_spread"]))
                    for r in rows if r["engine"] == engine and r["regime"] == regime
                    and to_float(r["ttft_p50_median"]) is not None]
            if not data:
                continue
            data.sort()
            concs = [d[0] for d in data]
            ttfts = [d[1] * 1000 for d in data]
            spreads = [(d[2] * 1000 if d[2] else 0) for d in data]
            lower = [max(0, t - s) for t, s in zip(ttfts, spreads)]
            upper = [t + s for t, s in zip(ttfts, spreads)]
            ax.plot(concs, ttfts, marker=MARKERS[engine], color=COLORS[engine],
                    label=engine, linewidth=2, markersize=7)
            ax.fill_between(concs, lower, upper, color=COLORS[engine], alpha=0.15)
        ax.set_xlabel("Concurrent Requests")
        ax.set_ylabel("TTFT p50 (ms)")
        ax.set_title(f"Time to First Token ({regime} regime)")
        ax.legend()
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 16, 32, 64])
        ax.set_xticklabels(["1", "4", "16", "32", "64"])
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "ttft_vs_concurrency.png")
    plt.savefig(path, dpi=150)
    plt.savefig(path.replace(".png", ".svg"))
    plt.close()
    print(f"  → {path}")


def plot_latency_p95_vs_concurrency(rows: list[dict], output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for idx, regime in enumerate(["short", "long"]):
        ax = axes[idx]
        for engine in ENGINES:
            data = [(int(r["concurrency"]), to_float(r["latency_p95_median"]), to_float(r["latency_p95_spread"]))
                    for r in rows if r["engine"] == engine and r["regime"] == regime
                    and to_float(r["latency_p95_median"]) is not None]
            if not data:
                continue
            data.sort()
            concs = [d[0] for d in data]
            lats = [d[1] * 1000 for d in data]
            spreads = [(d[2] * 1000 if d[2] else 0) for d in data]
            lower = [max(0, t - s) for t, s in zip(lats, spreads)]
            upper = [t + s for t, s in zip(lats, spreads)]
            ax.plot(concs, lats, marker=MARKERS[engine], color=COLORS[engine],
                    label=engine, linewidth=2, markersize=7)
            ax.fill_between(concs, lower, upper, color=COLORS[engine], alpha=0.15)
        ax.set_xlabel("Concurrent Requests")
        ax.set_ylabel("E2E Latency p95 (ms)")
        ax.set_title(f"End-to-End Latency p95 ({regime} regime)")
        ax.legend()
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 16, 32, 64])
        ax.set_xticklabels(["1", "4", "16", "32", "64"])
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "latency_p95_vs_concurrency.png")
    plt.savefig(path, dpi=150)
    plt.savefig(path.replace(".png", ".svg"))
    plt.close()
    print(f"  → {path}")


def plot_comparison_table(rows: list[dict], output_dir: str):
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.axis("off")

    cols = ["Engine", "Regime", "Conc", "Thru (tok/s)", "TTFT p50 (ms)",
            "TPOT p50 (ms)", "Lat p95 (ms)", "Lat p99 (ms)", "Success %"]
    table_data = []
    for r in rows:
        t = to_float(r.get("throughput_median"))
        ttft = to_float(r.get("ttft_p50_median"))
        tpot = to_float(r.get("tpot_p50_median"))
        lat95 = to_float(r.get("latency_p95_median"))
        lat99 = to_float(r.get("latency_p99_median"))
        sr = to_float(r.get("success_rate_median"))
        table_data.append([
            r["engine"],
            r["regime"],
            r["concurrency"],
            f"{t:.1f}" if t else "N/A",
            f"{ttft*1000:.1f}" if ttft else "N/A",
            f"{tpot*1000:.1f}" if tpot else "N/A",
            f"{lat95*1000:.1f}" if lat95 else "N/A",
            f"{lat99*1000:.1f}" if lat99 else "N/A",
            f"{sr*100:.1f}%" if sr else "N/A",
        ])

    table = ax.table(cellText=table_data, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    for i, label in enumerate(cols):
        table[0, i].set_facecolor("#f1f5f9")
        table[0, i].set_text_props(fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "comparison_table.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path}")


if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    csv_path = os.path.join(results_dir, "summary.csv")
    plots_dir = os.path.join(results_dir, "plots")

    if not os.path.exists(csv_path):
        print(f"No summary.csv at {csv_path}")
        print("Run: python scripts/collect_metrics.py first")
        sys.exit(1)

    os.makedirs(plots_dir, exist_ok=True)
    rows = load_csv(csv_path)
    print(f"Loaded {len(rows)} rows from {csv_path}")

    print("Generating plots...")
    plot_throughput_vs_concurrency(rows, plots_dir)
    plot_ttft_vs_concurrency(rows, plots_dir)
    plot_latency_p95_vs_concurrency(rows, plots_dir)
    plot_comparison_table(rows, plots_dir)
    print("Done.")
