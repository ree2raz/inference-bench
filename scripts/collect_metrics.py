"""Parse raw JSONL results into summary.csv and print stats."""
import json
import csv
import os
import sys
from collections import defaultdict
import statistics


def load_results(path: str) -> list[dict]:
    results = []
    if not os.path.exists(path):
        # try loading individual run JSONL files
        raw_dir = os.path.join(os.path.dirname(path), "raw")
        if os.path.isdir(raw_dir):
            for fname in sorted(os.listdir(raw_dir)):
                if fname.endswith(".jsonl"):
                    with open(os.path.join(raw_dir, fname)) as f:
                        for line in f:
                            if line.strip():
                                results.append(json.loads(line))
            return results
        return results
    with open(path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def aggregate_results(results: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in results:
        if "error" in r and "engine" not in r:
            continue
        key = (r["engine"], r["regime"], r["concurrency"])
        groups[key].append(r)

    rows = []
    for (engine, regime, conc), runs in sorted(groups.items()):
        if not runs:
            continue
        throughput = [r["throughput_tokens_per_sec"] for r in runs if r.get("throughput_tokens_per_sec")]
        ttft_p50 = [r["ttft_p50"] for r in runs if r.get("ttft_p50") is not None]
        ttft_p95 = [r["ttft_p95"] for r in runs if r.get("ttft_p95") is not None]
        tpot_p50 = [r["tpot_p50"] for r in runs if r.get("tpot_p50") is not None]
        tpot_p95 = [r["tpot_p95"] for r in runs if r.get("tpot_p95") is not None]
        lat_p50 = [r["latency_p50"] for r in runs if r.get("latency_p50") is not None]
        lat_p95 = [r["latency_p95"] for r in runs if r.get("latency_p95") is not None]
        lat_p99 = [r["latency_p99"] for r in runs if r.get("latency_p99") is not None]
        success_rate = [r["successful_requests"] / r["total_requests"] for r in runs if r.get("total_requests")]
        per_req = [r["throughput_per_request"] for r in runs if r.get("throughput_per_request")]

        def _med(vals):
            return round(statistics.median(vals), 4) if vals else None

        def _spread(vals):
            if len(vals) < 2:
                return None
            return round(max(vals) - min(vals), 4)

        rows.append({
            "engine": engine,
            "regime": regime,
            "concurrency": conc,
            "num_runs": len(runs),
            "throughput_median": _med(throughput),
            "throughput_spread": _spread(throughput),
            "throughput_per_request_median": _med(per_req),
            "ttft_p50_median": _med(ttft_p50),
            "ttft_p95_median": _med(ttft_p95),
            "ttft_p95_spread": _spread(ttft_p95),
            "tpot_p50_median": _med(tpot_p50),
            "tpot_p95_median": _med(tpot_p95),
            "latency_p50_median": _med(lat_p50),
            "latency_p95_median": _med(lat_p95),
            "latency_p99_median": _med(lat_p99),
            "latency_p95_spread": _spread(lat_p95),
            "success_rate_median": _med(success_rate),
        })
    return rows


def write_csv(rows: list[dict], path: str):
    if not rows:
        print("No rows to write.")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows → {path}")


def print_summary(rows: list[dict]):
    print(f"\n{'engine':<12} {'regime':<8} {'conc':>4} {'tok/s':>8} {'ttft_p50':>10} {'lat_p95':>10} {'success':>8}")
    print("-" * 70)
    for r in rows:
        t = r.get("throughput_median", "N/A")
        ttft = r.get("ttft_p50_median", "N/A")
        lat = r.get("latency_p95_median", "N/A")
        sr = r.get("success_rate_median", "N/A")
        t_str = f"{t:.1f}" if isinstance(t, (int, float)) else str(t)
        ttft_str = f"{ttft:.3f}" if isinstance(ttft, (int, float)) else str(ttft)
        lat_str = f"{lat:.3f}" if isinstance(lat, (int, float)) else str(lat)
        sr_str = f"{sr:.2f}" if isinstance(sr, (int, float)) else str(sr)
        print(f"{r['engine']:<12} {r['regime']:<8} {r['concurrency']:>4} {t_str:>8} {ttft_str:>10} {lat_str:>10} {sr_str:>8}")


if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    summary_jsonl = os.path.join(results_dir, "summary.jsonl")
    output_csv = os.path.join(results_dir, "summary.csv")

    results = load_results(summary_jsonl)
    if not results:
        print(f"No results found at {summary_jsonl}")
        print("Run benchmarks first: modal run modal_app.py")
        sys.exit(1)

    rows = aggregate_results(results)
    write_csv(rows, output_csv)
    print_summary(rows)
