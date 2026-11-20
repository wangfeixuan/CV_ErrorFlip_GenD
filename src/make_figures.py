from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_LABELS = {
    "base": "GenD",
    "clean_only": "Clean-only",
    "uniform": "Adapter-uniform",
    "flip_weighted": "Error-weighted",
    "flip_hard": "Hard-mask",
}
METHOD_COLORS = {
    "base": "#4D4D4D",
    "clean_only": "#7B61A8",
    "uniform": "#2E86AB",
    "flip_weighted": "#F18F01",
    "flip_hard": "#8D6E63",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "figure.dpi": 160,
        }
    )


def plot_local_accuracy(summary: pd.DataFrame, output: Path) -> None:
    conditions = list(dict.fromkeys(summary["condition"]))
    methods = ["base", "clean_only", "uniform", "flip_weighted", "flip_hard"]
    figure, axis = plt.subplots(figsize=(8.2, 4.3))
    width = 0.18
    for method_index, method in enumerate(methods):
        means, errors = [], []
        for condition in conditions:
            if method == "base":
                row = summary[(summary.condition == condition)].iloc[0]
            else:
                selected = summary[
                    (summary.condition == condition) & (summary.method == method)
                ]
                row = selected.iloc[0]
            means.append(100 * row["adapter_local_accuracy_mean"] if method != "base" else 100 * row["base_local_accuracy_mean"])
            errors.append(100 * row["adapter_local_accuracy_std"] if method != "base" else 0)
        positions = np.arange(len(conditions)) + (method_index - 2.0) * width
        axis.bar(
            positions,
            means,
            width,
            yerr=errors,
            capsize=2.5,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    axis.set_xticks(np.arange(len(conditions)))
    axis.set_xticklabels(["Blur k9 (train)", "JPEG q30 (unseen)", "Resize 25% (unseen)"])
    axis.set_ylabel("Video accuracy (%)")
    axis.set_ylim(75, 96)
    axis.set_title("Average accuracy over the six two-frame degradation masks")
    axis.legend(frameon=False, ncol=2, loc="upper left")
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def mask_curves(runs: Path, condition: str) -> pd.DataFrame:
    records = []
    for path in sorted(runs.glob("*/eval_test*.per_video.csv")):
        result_path = path.with_name(path.name.replace(".per_video.csv", ".json"))
        import json

        result = json.loads(result_path.read_text(encoding="utf-8"))
        current_condition = result.get("condition")
        if not current_condition:
            current_condition = (
                "blur_k9" if result_path.name == "eval_test.json" else ""
            )
        if current_condition != condition:
            continue
        method = result["method"]
        seed = result["seed"]
        frame = pd.read_csv(path, dtype={"mask_bits": str})
        frame["mask_bits"] = frame["mask_bits"].astype(str).str.zfill(4)
        frame = frame[frame["mask_bits"].str.count("1") == 2].copy()
        frame["method"] = method
        frame["seed"] = seed
        frame["target"] = frame["label"].eq("fake")
        frame["base_correct"] = (
            (frame["base_probability"] >= 0.5) == frame["target"]
        )
        frame["adapter_correct"] = (
            (frame["adapter_probability"] >= 0.5) == frame["target"]
        )
        records.append(frame)
    return pd.concat(records, ignore_index=True)


def plot_mask_curves(runs: Path, output: Path) -> None:
    frame = mask_curves(runs, "blur_k9")
    methods = ["base", "clean_only", "uniform", "flip_weighted", "flip_hard"]
    figure, axis = plt.subplots(figsize=(7.6, 3.8))
    mask_labels = None
    for method in methods:
        if method == "base":
            selected = frame
            column = "base_correct"
        else:
            selected = frame[frame["method"] == method]
            column = "adapter_correct"
        grouped = selected.groupby("mask_id", sort=True)[column].mean()
        labels = (
            selected.drop_duplicates("mask_id")
            .sort_values("mask_id")["mask_bits"]
            .tolist()
        )
        mask_labels = labels
        axis.plot(
            range(len(grouped)),
            100 * grouped.values,
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    axis.set_xticks(range(len(mask_labels)))
    axis.set_xticklabels(mask_labels, fontfamily="DejaVu Sans Mono")
    axis.set_xlabel("Blur mask, one character per frame")
    axis.set_ylabel("Test accuracy (%)")
    axis.set_ylim(74, 97)
    axis.set_title("Position sensitivity under exactly two blurred frames")
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def plot_recovery_harm(summary: pd.DataFrame, output: Path) -> None:
    selected = summary[
        summary.method.isin(["clean_only", "uniform", "flip_weighted", "flip_hard"])
    ]
    conditions = list(dict.fromkeys(selected.condition))
    methods = ["clean_only", "uniform", "flip_weighted", "flip_hard"]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), sharey=True)
    width = 0.24
    for method_index, method in enumerate(methods):
        subset = selected[selected.method == method]
        recovery = [
            100 * subset[subset.condition == condition].iloc[0]["recovery_rate_mean"]
            for condition in conditions
        ]
        harm = [
            100 * subset[subset.condition == condition].iloc[0]["harm_rate_mean"]
            for condition in conditions
        ]
        positions = np.arange(len(conditions)) + (method_index - 1.5) * width
        axes[0].bar(
            positions, recovery, width, color=METHOD_COLORS[method], label=METHOD_LABELS[method]
        )
        axes[1].bar(positions, harm, width, color=METHOD_COLORS[method])
    for axis, title in zip(axes, ["Base-error recovery", "Newly introduced errors"]):
        axis.set_xticks(np.arange(len(conditions)))
        axis.set_xticklabels(["Blur", "JPEG", "Resize"])
        axis.set_title(title)
        axis.set_ylim(0, 78)
    axes[0].set_ylabel("Rate (%)")
    axes[0].legend(frameon=False, loc="upper left")
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    runs = Path(args.runs_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(runs / "aggregate_all_conditions.summary.csv")
    apply_style()
    plot_local_accuracy(summary, output / "local_accuracy_by_condition.png")
    plot_mask_curves(runs, output / "position_sensitivity.png")
    plot_recovery_harm(summary, output / "recovery_and_harm.png")
    print(f"Saved figures to {output}")


if __name__ == "__main__":
    main()
