from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONDITION_LABELS = {
    "blur_k9": "Blur",
    "jpeg_q30": "JPEG",
    "resize_25": "Resize",
}
METHOD_LABELS = {
    "GenD": "Frozen GenD",
    "Uniform": "Uniform",
    "Restoration": "Feature Rest.",
    "Clean-only": "Clean-only",
}
COLORS = ("#24476B", "#2A9D8F", "#E76F51", "#E9C46A", "#7B6D8D")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2-dir", required=True)
    parser.add_argument("--core-dir", required=True)
    parser.add_argument("--mechanism-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style_axis(ax) -> None:
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_adapter_comparison(rows: list[dict], output: Path) -> None:
    conditions = list(CONDITION_LABELS)
    methods = ("GenD", "Uniform", "Restoration", "Clean-only")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), dpi=220)
    x = np.arange(len(conditions))
    width = 0.19
    for method_index, method in enumerate(methods):
        values = [
            next(
                float(row["accuracy_mean"])
                for row in rows
                if row["condition"] == condition
                and row["method"] == method
            )
            for condition in conditions
        ]
        axes[0].bar(
            x + (method_index - 1.5) * width,
            np.asarray(values) * 100,
            width,
            label=METHOD_LABELS[method],
            color=COLORS[method_index],
        )
        values = [
            next(
                float(row["balanced_accuracy_mean"])
                for row in rows
                if row["condition"] == condition
                and row["method"] == method
            )
            for condition in conditions
        ]
        axes[1].bar(
            x + (method_index - 1.5) * width,
            np.asarray(values) * 100,
            width,
            color=COLORS[method_index],
        )
    for ax, title in zip(
        axes, ("Video accuracy", "Balanced accuracy")
    ):
        ax.set_xticks(x, [CONDITION_LABELS[name] for name in conditions])
        ax.set_ylim(60, 100)
        ax.set_ylabel(f"{title} (%)")
        ax.set_title(title, fontsize=10)
        style_axis(ax)
    axes[0].legend(frameon=False, fontsize=7, ncol=2, loc="lower left")
    fig.suptitle(
        "Same four-frame all-degraded protocol: Adapter mechanisms",
        fontsize=11,
    )
    fig.tight_layout()
    save(fig, output)


def plot_residual_targets(rows: list[dict], output: Path) -> None:
    conditions = list(CONDITION_LABELS)
    residuals = ("R2", "R3")
    fig, ax = plt.subplots(figsize=(6.6, 3.4), dpi=220)
    x = np.arange(len(conditions))
    width = 0.32
    for index, residual in enumerate(residuals):
        values = [
            next(
                float(row["balanced_accuracy"])
                for row in rows
                if row["condition"] == condition
                and row["residual"] == residual
                and row["start"] == "za"
                and row["head"] == "adapter"
                and row["selected"] == "True"
            )
            for condition in conditions
        ]
        ax.bar(
            x + (index - 0.5) * width,
            np.asarray(values) * 100,
            width,
            label=residual,
            color=COLORS[index],
        )
        for position, value in zip(x, values):
            ax.text(
                position + (index - 0.5) * width,
                value * 100 + 0.5,
                f"{value * 100:.1f}",
                ha="center",
                fontsize=7,
            )
    ax.set_xticks(x, [CONDITION_LABELS[name] for name in conditions])
    ax.set_ylim(80, 100)
    ax.set_ylabel("Balanced accuracy (%)")
    ax.set_title(
        "R3 improves blur and resize; JPEG is close", fontsize=10
    )
    ax.legend(frameon=False)
    style_axis(ax)
    fig.tight_layout()
    save(fig, output)


def plot_interpolation(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0), dpi=220, sharey=True)
    for ax, condition in zip(axes, CONDITION_LABELS):
        for index, residual in enumerate(("R1", "R2", "R3")):
            subset = [
                row
                for row in rows
                if row["condition"] == condition
                and row["residual"] == residual
                and row["head"] == "adapter"
            ]
            if not subset:
                continue
            subset.sort(key=lambda row: float(row["alpha"]))
            ax.plot(
                [float(row["alpha"]) for row in subset],
                [float(row["balanced_accuracy"]) * 100 for row in subset],
                marker="o",
                markersize=3,
                linewidth=1.6,
                label=residual,
                color=COLORS[index],
            )
        ax.set_title(CONDITION_LABELS[condition], fontsize=9)
        ax.set_xlabel("Residual coefficient α")
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        style_axis(ax)
    axes[0].set_ylabel("Balanced accuracy (%)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Whole-residual interpolation baseline", fontsize=10)
    fig.tight_layout()
    save(fig, output)


def plot_selectors(
    core_rows: list[dict],
    random_rows: list[dict],
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5), dpi=220, sharey=True)
    selectors = (
        "random",
        "magnitude",
        "classifier",
        "signed_discriminative",
        "discriminative",
    )
    for ax, condition in zip(axes, CONDITION_LABELS):
        subset = [
            row
            for row in core_rows
            if row["condition"] == condition
            and row["residual"] == "R3"
            and row["head"] == "adapter"
        ]
        random_subset = [
            row
            for row in random_rows
            if row["condition"] == condition
            and row["residual"] == "R3"
            and row["split"] == "test"
        ]
        values = [
            next(
                float(row["add_balanced_accuracy"])
                for row in subset
                if row["selector"] == selector
            )
            * 100
            for selector in selectors
        ]
        errors = []
        control_means = []
        for selector in selectors:
            match = next(
                row
                for row in random_subset
                if row["selector"] == selector
            )
            errors.append(
                float(match["random_balanced_accuracy_std"]) * 100
            )
            control_means.append(
                float(match["random_balanced_accuracy_mean"]) * 100
            )
        positions = np.arange(len(selectors))
        ax.bar(
            positions,
            values,
            yerr=errors,
            capsize=3,
            color=["#9CA3AF", "#2A9D8F", "#457B9D", "#E76F51", "#7B6D8D"],
        )
        ax.scatter(
            positions,
            control_means,
            marker="_",
            s=220,
            linewidths=2,
            color="#111827",
            zorder=4,
            label="Matched random mean",
        )
        ax.set_xticks(
            positions,
            ("Random", "Magnitude", "Classifier", "Signed", "ReLU"),
            rotation=28,
            ha="right",
            fontsize=7,
        )
        ax.set_title(CONDITION_LABELS[condition], fontsize=9)
        ax.set_ylim(80, 100)
        style_axis(ax)
    axes[0].set_ylabel("Balanced accuracy (%)")
    fig.suptitle(
        "R3 selectors (bars) vs energy-matched random controls (black bars)",
        fontsize=10,
    )
    axes[0].legend(frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout()
    save(fig, output)


def plot_subspaces(rows: list[dict], output: Path) -> None:
    selectors = (
        "pca16_parallel",
        "pca32_parallel",
        "discriminant_parallel",
        "discriminant_orthogonal",
    )
    labels = ("PCA-16", "PCA-32", "Direction", "Orthogonal")
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2), dpi=220, sharey=True)
    for ax, condition in zip(axes, CONDITION_LABELS):
        values = []
        for selector in selectors:
            match = next(
                (
                    row
                    for row in rows
                    if row["condition"] == condition
                    and row["residual"] == "R3"
                    and row["selector"] == selector
                ),
                None,
            )
            values.append(
                float(match["balanced_accuracy"]) * 100 if match else np.nan
            )
        positions = np.arange(len(selectors))
        ax.bar(positions, values, color=COLORS[:4])
        ax.set_xticks(positions, labels, rotation=20, ha="right")
        ax.set_title(CONDITION_LABELS[condition], fontsize=9)
        ax.set_ylim(80, 100)
        style_axis(ax)
    axes[0].set_ylabel("Balanced accuracy (%)")
    fig.suptitle("R3 low-dimensional subspace compensation", fontsize=10)
    fig.tight_layout()
    save(fig, output)


def plot_ridge(rows: list[dict], output: Path) -> None:
    predictors = (
        "zero_residual",
        "training_mean_residual",
        "ridge_full_residual",
        "ridge_subspace_residual",
        "oracle_true_clean_endpoint",
    )
    labels = ("Zero", "Mean", "Ridge full", "Ridge 32D", "Oracle")
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), dpi=220, sharey=True)
    for ax, condition in zip(axes, CONDITION_LABELS):
        values = []
        for predictor in predictors:
            match = next(
                row
                for row in rows
                if row["condition"] == condition
                and row["input"] == "za"
                and row["predictor"] == predictor
            )
            values.append(float(match["balanced_accuracy"]) * 100)
        positions = np.arange(len(predictors))
        ax.bar(
            positions,
            values,
            color=("#9CA3AF", "#B8C4CE", "#457B9D", "#2A9D8F", "#E76F51"),
        )
        ax.set_xticks(positions, labels, rotation=25, ha="right", fontsize=7)
        ax.set_title(CONDITION_LABELS[condition], fontsize=9)
        ax.set_ylim(80, 100)
        style_axis(ax)
    axes[0].set_ylabel("Balanced accuracy (%)")
    fig.suptitle(
        "Deployable residual predictability vs reference endpoints",
        fontsize=10,
    )
    fig.tight_layout()
    save(fig, output)


def plot_error_groups(rows: list[dict], output: Path) -> None:
    names = (
        "A_successful_correction",
        "B_persistent_degradation_error",
        "C_adapter_introduced_error",
        "D_persistent_original_error",
        "E_stable_correct",
    )
    labels = ("A corrected", "B persistent", "C harmed", "D base error", "E stable")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.5), dpi=220)
    x = np.arange(len(names))
    width = 0.25
    for index, condition in enumerate(CONDITION_LABELS):
        subset = {
            row["group"]: row
            for row in rows
            if row["condition"] == condition and row["split"] == "test"
        }
        axes[0].bar(
            x + (index - 1) * width,
            [
                float(subset[name]["videos_mean"])
                if name in subset
                else 0.0
                for name in names
            ],
            width,
            label=CONDITION_LABELS[condition],
            color=COLORS[index],
        )
        axes[1].bar(
            x + (index - 1) * width,
            [
                float(subset[name]["R3_norm_mean"])
                if name in subset
                else np.nan
                for name in names
            ],
            width,
            color=COLORS[index],
        )
    axes[0].set_xticks(x, labels, rotation=20, ha="right", fontsize=7)
    axes[0].set_ylabel("Test videos")
    axes[0].set_title("Error-transition group size", fontsize=9)
    axes[1].set_xticks(x, labels, rotation=20, ha="right", fontsize=7)
    axes[1].set_ylabel("Mean R3 norm")
    axes[1].set_title("Within-group R3 residual", fontsize=9)
    for ax in axes:
        style_axis(ax)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, output)


def plot_cross_heads(rows: list[dict], output: Path) -> None:
    features = ("z0", "zd", "zu", "zf")
    labels = ("Clean z0", "Deg. zd", "Uniform zu", "Restore zf")
    heads = ("original", "uniform", "restoration")
    head_labels = ("GenD head", "Uniform head", "Restore head")
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), dpi=220, sharey=True)
    for ax, condition in zip(axes, CONDITION_LABELS):
        x = np.arange(len(features))
        width = 0.25
        for index, (head, head_label) in enumerate(zip(heads, head_labels)):
            values = [
                float(
                    next(
                        row["balanced_accuracy_mean"]
                        for row in rows
                        if row["condition"] == condition
                        and row["split"] == "test"
                        and row["head"] == head
                        and row["features"] == feature
                    )
                )
                * 100
                for feature in features
            ]
            ax.bar(
                x + (index - 1) * width,
                values,
                width,
                label=head_label,
                color=COLORS[index],
            )
        ax.set_xticks(x, labels, rotation=24, ha="right", fontsize=7)
        ax.set_title(CONDITION_LABELS[condition], fontsize=9)
        ax.set_ylim(60, 100)
        style_axis(ax)
    axes[0].set_ylabel("Balanced accuracy (%)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Feature × frozen-head cross diagnosis", fontsize=10)
    fig.tight_layout()
    save(fig, output)


def plot_bootstrap(rows: list[dict], output: Path) -> None:
    rows = [
        row
        for row in rows
        if row["comparison"] != "all_three_paired"
        and row["condition"] in CONDITION_LABELS
        and row.get("ci95_lower")
    ]
    fig, ax = plt.subplots(figsize=(7.8, 4.0), dpi=220)
    labels = []
    for index, row in enumerate(rows):
        value = float(row["difference"]) * 100
        low = float(row["ci95_lower"]) * 100
        high = float(row["ci95_upper"]) * 100
        color = "#2A9D8F" if low > 0 else "#8D99AE"
        ax.errorbar(
            value,
            index,
            xerr=[[value - low], [high - value]],
            fmt="o",
            color=color,
            capsize=3,
        )
        labels.append(
            f"{CONDITION_LABELS[row['condition']]} · "
            f"{row['comparison'].replace('_', ' ')}"
        )
    ax.set_yticks(np.arange(len(rows)), labels, fontsize=6)
    ax.axvline(0, color="#6B7280", linewidth=0.8)
    ax.set_xlabel("Paired source-group accuracy difference (pp)")
    ax.set_title("Exploratory paired bootstrap, 95% intervals", fontsize=10)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    ax.invert_yaxis()
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    save(fig, output)


def plot_calibration(rows: list[dict], output: Path) -> None:
    conditions = list(CONDITION_LABELS)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), dpi=220)
    x = np.arange(len(conditions))
    width = 0.24
    for index, model in enumerate(("GenD", "Adapter")):
        raw = [
            next(
                float(row["raw_balanced_accuracy"])
                for row in rows
                if row["condition"] == condition and row["model"] == model
            )
            * 100
            for condition in conditions
        ]
        calibrated = [
            next(
                float(row["calibrated_balanced_accuracy"])
                for row in rows
                if row["condition"] == condition and row["model"] == model
            )
            * 100
            for condition in conditions
        ]
        axes[0].bar(
            x + (2 * index - 1.5) * width,
            raw,
            width,
            label=f"{model} raw",
            color=COLORS[index],
        )
        axes[0].bar(
            x + (2 * index - 0.5) * width,
            calibrated,
            width,
            label=f"{model} calibrated",
            color=COLORS[index],
            alpha=0.55,
        )
        raw_nll = [
            next(
                float(row["raw_nll"])
                for row in rows
                if row["condition"] == condition and row["model"] == model
            )
            for condition in conditions
        ]
        calibrated_nll = [
            next(
                float(row["calibrated_nll"])
                for row in rows
                if row["condition"] == condition and row["model"] == model
            )
            for condition in conditions
        ]
        axes[1].bar(
            x + (2 * index - 1.5) * width,
            raw_nll,
            width,
            color=COLORS[index],
        )
        axes[1].bar(
            x + (2 * index - 0.5) * width,
            calibrated_nll,
            width,
            color=COLORS[index],
            alpha=0.55,
        )
    disagreement = [
        next(
            float(row["frame_video_rule_disagreement"])
            for row in rows
            if row["condition"] == condition and row["model"] == "Adapter"
        )
        * 100
        for condition in conditions
    ]
    axes[2].bar(x, disagreement, color=COLORS[2])
    for ax, title in zip(
        axes,
        ("Balanced accuracy (%)", "Video-level NLL", "Frame/video disagreement (%)"),
    ):
        ax.set_xticks(x, [CONDITION_LABELS[name] for name in conditions])
        ax.set_title(title, fontsize=8)
        style_axis(ax)
    axes[0].set_ylim(70, 95)
    axes[0].legend(frameon=False, fontsize=6)
    fig.suptitle("Validation-fitted temperature and video-level decision audit", fontsize=10)
    fig.tight_layout()
    save(fig, output)


def main() -> None:
    args = parse_args()
    stage2_dir = Path(args.stage2_dir)
    core_dir = Path(args.core_dir)
    mechanism_dir = Path(args.mechanism_dir)
    output_dir = Path(args.output_dir)
    plot_adapter_comparison(
        read_csv(mechanism_dir / "adapter_summary.csv"),
        output_dir / "adapter_mechanism_comparison.png",
    )
    plot_residual_targets(
        read_csv(core_dir / "interpolation_test.csv"),
        output_dir / "residual_target_comparison.png",
    )
    plot_interpolation(
        read_csv(core_dir / "interpolation_test.csv"),
        output_dir / "whole_residual_interpolation.png",
    )
    plot_selectors(
        read_csv(core_dir / "selection_test.csv"),
        read_csv(stage2_dir / "matched_random_test.csv"),
        output_dir / "selector_vs_matched_random.png",
    )
    plot_subspaces(
        read_csv(stage2_dir / "subspace_test.csv"),
        output_dir / "subspace_comparison.png",
    )
    plot_ridge(
        read_csv(stage2_dir / "ridge_predictability.csv"),
        output_dir / "ridge_predictability.png",
    )
    plot_error_groups(
        read_csv(mechanism_dir / "error_groups_summary.csv"),
        output_dir / "error_conditioned_residuals.png",
    )
    plot_cross_heads(
        read_csv(mechanism_dir / "cross_head_summary.csv"),
        output_dir / "feature_head_cross_diagnosis.png",
    )
    plot_bootstrap(
        read_csv(stage2_dir / "bootstrap.csv"),
        output_dir / "paired_bootstrap_intervals.png",
    )
    plot_calibration(
        read_csv(core_dir / "calibration.csv"),
        output_dir / "calibration_and_video_decision.png",
    )
    print(output_dir)


if __name__ == "__main__":
    main()
