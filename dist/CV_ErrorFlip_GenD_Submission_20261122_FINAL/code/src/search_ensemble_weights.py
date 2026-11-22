from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import binary_metrics


EXPERTS = {
    "clean": "clean_only",
    "blur": "uniform",
    "jpeg": "jpeg_only",
    "resize": "resize_only",
}
CONDITIONS = {
    "blur_k9": ("eval_{split}.json", "blur_k9"),
    "jpeg_q30": ("eval_{split}_jpeg_q30.json", "jpeg_q30"),
    "resize_25": ("eval_{split}_resize_25.json", "resize_25"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--blur-step", type=float, default=0.1)
    parser.add_argument("--clean-weights", default="0,0.1,0.2,0.3")
    return parser.parse_args()


def load_seed_matrices(
    runs: Path,
    split: str,
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray, list[str]]:
    result: dict[str, dict[str, np.ndarray]] = {
        condition: {} for condition in CONDITIONS
    }
    labels = None
    video_ids = None
    for seed in (2026, 2027, 2028):
        seed_experts: dict[str, list[np.ndarray]] = {
            expert: [] for expert in EXPERTS
        }
        for condition, (filename_template, suffix) in CONDITIONS.items():
            filename = filename_template.format(split=split)
            expert_arrays = []
            for expert_key, method in EXPERTS.items():
                run_dir = runs / f"{method}_seed{seed}"
                csv_path = run_dir / filename.replace(".json", ".per_video.csv")
                frame = pd.read_csv(csv_path, dtype={"mask_bits": str})
                frame["mask_bits"] = frame["mask_bits"].astype(str).str.zfill(4)
                current_ids = sorted(frame["video_id"].unique())
                mask_ids = sorted(frame["mask_id"].unique())
                index = {video_id: i for i, video_id in enumerate(current_ids)}
                mask_index = {mask_id: i for i, mask_id in enumerate(mask_ids)}
                matrix = np.zeros((len(current_ids), len(mask_ids)), dtype=float)
                for row in frame.itertuples(index=False):
                    matrix[index[row.video_id], mask_index[row.mask_id]] = (
                        row.adapter_probability
                    )
                if video_ids is None:
                    video_ids = current_ids
                if current_ids != video_ids:
                    raise ValueError(f"Video order mismatch in {csv_path}")
                expert_arrays.append(matrix)
            seed_experts[condition] = expert_arrays
            if labels is None:
                label_map = (
                    frame[["video_id", "label"]]
                    .drop_duplicates()
                    .set_index("video_id")["label"]
                )
                labels = np.asarray(
                    [label_map[video_id] == "fake" for video_id in current_ids],
                    dtype=bool,
                )
        for condition in CONDITIONS:
            for expert_index, expert_key in enumerate(EXPERTS):
                result[condition].setdefault(expert_key, []).append(
                    seed_experts[condition][expert_index]
                )
    averaged = {
        condition: {
            expert: np.mean(seed_values, axis=0)
            for expert, seed_values in expert_values.items()
        }
        for condition, expert_values in result.items()
    }
    return averaged, labels, video_ids


def combined_probability(
    matrices: dict[str, np.ndarray], weights: dict[str, float]
) -> np.ndarray:
    result = np.zeros_like(next(iter(matrices.values())))
    for expert, weight in weights.items():
        result += weight * matrices[expert]
    return result


def condition_metric(
    probabilities: np.ndarray,
    labels: np.ndarray,
    select_pair_two: bool,
    threshold: float,
) -> dict:
    if select_pair_two:
        # Use explicit mask IDs whose binary representation contains two ones.
        mask_ids = [
            mask_id for mask_id in range(16) if f"{mask_id:04b}".count("1") == 2
        ]
    else:
        mask_ids = [0]
    metrics = [
        binary_metrics(labels, probabilities[:, mask_id], threshold)
        for mask_id in mask_ids
    ]
    return {
        "balanced_accuracy": float(
            np.mean([metric["balanced_accuracy"] for metric in metrics])
        ),
        "accuracy": float(np.mean([metric["accuracy"] for metric in metrics])),
        "auc": float(np.mean([metric["auc"] for metric in metrics])),
    }


def score_weights(
    matrices: dict[str, dict[str, np.ndarray]],
    labels: np.ndarray,
    weights: dict[str, float],
    threshold: float,
) -> dict:
    metrics = {}
    for condition in CONDITIONS:
        metrics[condition] = condition_metric(
            combined_probability(matrices[condition], weights),
            labels,
            select_pair_two=True,
            threshold=threshold,
        )
    # The clean input is shared conceptually; use the blur cache's mask 0000.
    metrics["clean"] = condition_metric(
        combined_probability(matrices["blur_k9"], weights),
        labels,
        select_pair_two=False,
        threshold=threshold,
    )
    metrics["objective"] = float(
        np.mean(
            [
                metrics["clean"]["balanced_accuracy"],
                metrics["blur_k9"]["balanced_accuracy"],
                metrics["jpeg_q30"]["balanced_accuracy"],
                metrics["resize_25"]["balanced_accuracy"],
            ]
        )
    )
    return metrics


def weight_grid(
    blur_step: float, clean_weights: list[float]
) -> list[dict[str, float]]:
    fractions = np.arange(0.0, 1.0 + blur_step / 2, blur_step)
    rows = []
    for clean_weight in clean_weights:
        degraded_weight = 1.0 - clean_weight
        for blur_fraction in fractions:
            rows.append(
                {
                    "clean": float(clean_weight),
                    "blur": float(degraded_weight * blur_fraction),
                    "jpeg": float(degraded_weight * (1 - blur_fraction) / 2),
                    "resize": float(degraded_weight * (1 - blur_fraction) / 2),
                }
            )
    return rows


def flatten_metrics(
    split: str, weights: dict[str, float], metrics: dict
) -> dict:
    row = {"split": split, **weights, "objective": metrics["objective"]}
    for condition in ("clean", *CONDITIONS):
        for key in ("balanced_accuracy", "accuracy", "auc"):
            row[f"{condition}_{key}"] = metrics[condition][key]
    return row


def main() -> None:
    args = parse_args()
    runs = Path(args.runs_dir)
    validation, labels, video_ids = load_seed_matrices(runs, "validation")
    test, test_labels, test_video_ids = load_seed_matrices(runs, "test")

    scan = []
    for weights in weight_grid(
        args.blur_step,
        [float(value) for value in args.clean_weights.split(",")],
    ):
        metrics = score_weights(validation, labels, weights, threshold=0.5)
        scan.append(flatten_metrics("validation", weights, metrics))
    best = max(scan, key=lambda row: (row["objective"], -row["clean"]))
    selected_weights = {expert: best[expert] for expert in EXPERTS}
    test_metrics = score_weights(test, test_labels, selected_weights, threshold=0.5)

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scan).to_csv(prefix.with_name(prefix.name + "_scan.csv"), index=False)
    summary = {
        "selected_weights": selected_weights,
        "validation": best,
        "test": flatten_metrics("test", selected_weights, test_metrics),
        "selected_on": "mean balanced accuracy over clean, blur, JPEG and resize",
    }
    prefix.with_name(prefix.name + "_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    frame = pd.DataFrame(scan)
    for clean_weight, group in frame.groupby("clean"):
        group = group.sort_values("blur")
        axis.plot(
            100 * group["blur"],
            100 * group["objective"],
            marker="o",
            linewidth=1.6,
            label=f"Clean expert: {100 * clean_weight:.0f}%",
        )
    axis.axvline(100 * selected_weights["blur"], color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("Blur expert weight (%)")
    axis.set_ylabel("Validation mean balanced accuracy (%)")
    axis.set_title("Static specialist-ensemble weight sweep")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(prefix.with_name(prefix.name + "_curve.png"), bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
