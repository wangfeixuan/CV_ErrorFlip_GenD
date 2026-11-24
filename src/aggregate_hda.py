from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = Path(args.runs_dir)
    rows = []
    for path in sorted(runs.glob("*/run.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        for suite, result in run["test"].items():
            rows.append(
                {
                    "method": run["method"],
                    "seed": int(run["seed"]),
                    "suite": suite,
                    **{
                        key: result[key]
                        for key in (
                            "base_mean_accuracy",
                            "base_worst_accuracy",
                            "adapter_mean_accuracy",
                            "adapter_worst_accuracy",
                            "base_oracle_mean_accuracy",
                            "base_oracle_worst_accuracy",
                            "adapter_oracle_mean_accuracy",
                            "adapter_oracle_worst_accuracy",
                            "base_clean",
                            "adapter_clean",
                            "mean_clean_weight",
                            "mean_degraded_weight",
                        )
                    },
                }
            )
            rows[-1]["base_clean_accuracy"] = rows[-1].pop("base_clean")["accuracy"]
            rows[-1]["adapter_clean_accuracy"] = rows[-1].pop("adapter_clean")[
                "accuracy"
            ]
    if not rows:
        raise FileNotFoundError("No heterogeneous-degradation runs found")
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_suffix(".per_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["suite"], row["method"])].append(row)
    summary = []
    metrics = [
        "base_mean_accuracy",
        "base_worst_accuracy",
        "adapter_mean_accuracy",
        "adapter_worst_accuracy",
        "base_oracle_mean_accuracy",
        "base_oracle_worst_accuracy",
        "adapter_oracle_mean_accuracy",
        "adapter_oracle_worst_accuracy",
        "base_clean_accuracy",
        "adapter_clean_accuracy",
        "mean_clean_weight",
        "mean_degraded_weight",
    ]
    for (suite, method), items in sorted(grouped.items()):
        aggregate = {"suite": suite, "method": method, "seeds": len(items)}
        for metric in metrics:
            values = [float(item[metric]) for item in items]
            aggregate[f"{metric}_mean"] = float(np.mean(values))
            aggregate[f"{metric}_std"] = float(np.std(values, ddof=0))
        summary.append(aggregate)
    with prefix.with_suffix(".summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    lines = [
        "# Heterogeneous degradation aggregation",
        "",
        "| Suite | Method | Seeds | Base clean | Adapter clean | Base mixed | Adapter mixed | Adapter worst | Base oracle | Adapter oracle | Clean weight | Degraded weight |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {suite} | {method} | {seeds} | {base_clean:.2f}% | "
            "{adapter_clean:.2f}% | {base:.2f}% | {adapter:.2f}% | "
            "{worst:.2f}% | {base_oracle:.2f}% | {adapter_oracle:.2f}% | "
            "{clean_w:.3f} | {deg_w:.3f} |".format(
                suite=row["suite"],
                method=row["method"],
                seeds=row["seeds"],
                base_clean=100 * row["base_clean_accuracy_mean"],
                adapter_clean=100 * row["adapter_clean_accuracy_mean"],
                base=100 * row["base_mean_accuracy_mean"],
                adapter=100 * row["adapter_mean_accuracy_mean"],
                worst=100 * row["adapter_worst_accuracy_mean"],
                base_oracle=100 * row["base_oracle_mean_accuracy_mean"],
                adapter_oracle=100
                * row["adapter_oracle_mean_accuracy_mean"],
                clean_w=row["mean_clean_weight_mean"],
                deg_w=row["mean_degraded_weight_mean"],
            )
        )
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
