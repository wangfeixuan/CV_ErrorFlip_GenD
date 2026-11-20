from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = Path(args.runs_dir)
    values: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for path in sorted(runs.glob("k*_random_seed*/eval_test*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        train_k = int(result.get("training_blurred_frames", 2))
        condition = result.get("condition", path.stem)
        for test_k, section in result.get("adapter_by_mask_count", {}).items():
            values[(condition, train_k, int(test_k))].append(
                float(section["mean_accuracy"])
            )
    if not values:
        raise ValueError("No k-sweep evaluations with adapter_by_mask_count found")

    rows = []
    for (condition, train_k, test_k), samples in sorted(values.items()):
        rows.append(
            {
                "condition": condition,
                "train_k": train_k,
                "test_k": test_k,
                "seeds": len(samples),
                "mean_accuracy": float(np.mean(samples)),
                "std_accuracy": float(np.std(samples)),
            }
        )
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    selected = [row for row in rows if row["condition"] == "blur_k9"]
    matrix = np.full((4, 5), np.nan)
    for row in selected:
        matrix[row["train_k"] - 1, row["test_k"]] = 100 * row["mean_accuracy"]

    figure, axis = plt.subplots(figsize=(6.8, 3.8))
    image = axis.imshow(matrix, cmap="YlGnBu", vmin=75, vmax=95, aspect="auto")
    for train_index in range(matrix.shape[0]):
        for test_k in range(matrix.shape[1]):
            axis.text(
                test_k,
                train_index,
                f"{matrix[train_index, test_k]:.1f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )
    axis.set_xticks(range(5))
    axis.set_xticklabels(["0\nclean", "1", "2", "3", "4"])
    axis.set_yticks(range(4))
    axis.set_yticklabels(["k=1", "k=2", "k=3", "k=4"])
    axis.set_xlabel("Blurred frames at evaluation")
    axis.set_ylabel("Blurred frames during training")
    axis.set_title("Mask-count sensitivity on held-out blur k9")
    figure.colorbar(image, ax=axis, label="Accuracy (%)", fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(prefix.with_suffix(".png"), bbox_inches="tight")
    plt.close(figure)
    print(prefix.with_suffix(".csv"))
    print(prefix.with_suffix(".png"))


if __name__ == "__main__":
    main()
