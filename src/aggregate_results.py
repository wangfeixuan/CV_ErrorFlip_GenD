from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = {
    "base_clean_accuracy": ("base_clean", "accuracy"),
    "base_clean_auc": ("base_clean", "auc"),
    "adapter_clean_accuracy": ("adapter_clean", "accuracy"),
    "adapter_clean_auc": ("adapter_clean", "auc"),
    "base_local_accuracy": ("base_two_frame_masks", "mean_accuracy"),
    "base_local_auc": ("base_two_frame_masks", "mean_auc"),
    "base_worst_accuracy": ("base_two_frame_masks", "worst_mask_accuracy"),
    "adapter_local_accuracy": ("adapter_two_frame_masks", "mean_accuracy"),
    "adapter_local_auc": ("adapter_two_frame_masks", "mean_auc"),
    "adapter_worst_accuracy": ("adapter_two_frame_masks", "worst_mask_accuracy"),
    "base_flip_rate": ("base_two_frame_masks", "flip_rate"),
    "adapter_flip_rate": ("adapter_two_frame_masks", "flip_rate"),
    "recovery_rate": ("inversion", "recovery_rate"),
    "harm_rate": ("inversion", "harm_rate"),
    "net_recovered_mask_events": ("inversion", "net_recovered_mask_events"),
    "base_flipped_videos": ("inversion", "base_flipped_videos"),
    "flipped_videos_with_any_recovery": (
        "inversion",
        "base_flipped_videos_with_any_recovery",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args()


def flatten(result: dict) -> dict:
    row = {
        "method": result["method"],
        "condition": result.get("condition", result.get("feature_file", "unknown")),
        "training_blurred_frames": int(
            result.get("training_blurred_frames", 2)
        ),
        "seed": int(result["seed"]),
        "test_videos": int(result["videos"]),
    }
    for output_name, (section, key) in METRICS.items():
        row[output_name] = result[section][key]
    return row


def mean_std(values) -> tuple[float | None, float | None]:
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return None, None
    return float(np.mean(finite)), float(np.std(finite, ddof=0))


def percent(value) -> str:
    return "n/a" if value is None else f"{100 * float(value):.2f}%"


def main() -> None:
    args = parse_args()
    runs = Path(args.runs_dir)
    rows = [
        flatten(json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(runs.glob("*/eval_test*.json"))
    ]
    if not rows:
        raise FileNotFoundError(f"No */eval_test*.json under {runs}")
    rows.sort(
        key=lambda row: (
            row["condition"],
            row["method"],
            row["training_blurred_frames"],
            row["seed"],
        )
    )

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_suffix(".per_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_method: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        by_method[
            (
                row["condition"],
                row["method"],
                row["training_blurred_frames"],
            )
        ].append(row)
    summary = []
    for (
        condition,
        method,
        training_blurred_frames,
    ), method_rows in sorted(by_method.items()):
        aggregate = {
            "condition": condition,
            "method": method,
            "training_blurred_frames": training_blurred_frames,
            "seeds": len(method_rows),
        }
        for metric in METRICS:
            average, standard_deviation = mean_std(row[metric] for row in method_rows)
            aggregate[f"{metric}_mean"] = average
            aggregate[f"{metric}_std"] = standard_deviation
        summary.append(aggregate)
    with prefix.with_suffix(".summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    lines = [
        "# Seed aggregation",
        "",
        f"Evaluation files: `{len(rows)}`; test videos per seed: `{rows[0]['test_videos']}`.",
        "",
        "| Condition | Method | Train k | Seeds | Clean Acc | Clean AUC | Local Acc | Local AUC | Worst Acc | Recovery | Harm | Net recovered masks |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {condition} | {method} | {train_k} | {seeds} | {clean_acc} | {clean_auc} | {local_acc} | "
            "{local_auc} | {worst} | {recovery} | {harm} | {net:.2f} |".format(
                condition=row["condition"],
                method=row["method"],
                train_k=row["training_blurred_frames"],
                seeds=row["seeds"],
                clean_acc=percent(row["adapter_clean_accuracy_mean"]),
                clean_auc=f"{row['adapter_clean_auc_mean']:.4f}",
                local_acc=percent(row["adapter_local_accuracy_mean"]),
                local_auc=f"{row['adapter_local_auc_mean']:.4f}",
                worst=percent(row["adapter_worst_accuracy_mean"]),
                recovery=percent(row["recovery_rate_mean"]),
                harm=percent(row["harm_rate_mean"]),
                net=row["net_recovered_mask_events_mean"],
            )
        )
    lines.extend(
        [
            "",
            "Results are test-split summaries after seed-level early stopping on validation.",
            "The 16 masks are repeated measurements of the same videos, not independent samples.",
            "Use the per-seed table and paired differences for the final report.",
            "",
        ]
    )
    prefix.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
