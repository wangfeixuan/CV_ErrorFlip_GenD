from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .adapter import GenDAdapter
from .gend_model import choose_device
from .metrics import binary_metrics


CONDITIONS = {
    "blur_k9": "Blur k9",
    "jpeg_q30": "JPEG q30",
    "resize_25": "Resize 25%",
}
RATIOS = (0.01, 0.05, 0.10, 0.20, 0.50)
SELECTORS = (
    "random",
    "magnitude",
    "classifier",
    "discriminative",
    "signed_discriminative",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline residual diagnostics for a frozen GenD Adapter."
    )
    parser.add_argument("--blur-features", required=True)
    parser.add_argument("--jpeg-features", required=True)
    parser.add_argument("--resize-features", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[GenDAdapter, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = GenDAdapter.from_pretrained_head(
        torch.from_numpy(checkpoint["head_weight"]),
        torch.from_numpy(checkpoint["head_bias"]),
        bottleneck=int(checkpoint["config"]["model"]["bottleneck"]),
        dropout=float(checkpoint["config"]["model"]["dropout"]),
        scale_init=float(checkpoint["config"]["model"]["scale_init"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


def split_indices(checkpoint: dict, split: str, video_ids: np.ndarray) -> np.ndarray:
    split_by_video = checkpoint["split_by_video"]
    return np.asarray(
        [
            index
            for index, video_id in enumerate(video_ids)
            if split_by_video[str(video_id)] == split
        ],
        dtype=np.int64,
    )


def correct_class_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    correct = logits.gather(-1, labels[:, None, None].expand(-1, logits.shape[1], 1))
    other = logits.gather(
        -1,
        (1 - labels)[:, None, None].expand(-1, logits.shape[1], 1),
    )
    return (correct - other).squeeze(-1).mean(dim=-1)


@torch.inference_mode()
def feature_state(
    model: GenDAdapter,
    data: dict,
    indices: np.ndarray,
    device: torch.device,
) -> dict:
    z0 = torch.from_numpy(data["features_original"][indices]).to(device).float()
    zd = torch.from_numpy(data["features_blur"][indices]).to(device).float()
    zr = model.adapter(zd)
    zc = model.adapter(z0)
    base_clean_logits = torch.from_numpy(data["base_logits"][indices]).to(device)
    base_degraded_logits = torch.from_numpy(data["base_logits_blur"][indices]).to(device)
    adapter_logits = model.head(zr)
    labels = torch.from_numpy(data["label_index"][indices].astype(np.int64)).to(device)
    return {
        "z0": z0,
        "zd": zd,
        "zr": zr,
        "zc": zc,
        "r_degradation": z0 - zd,
        "r_recovered": zr - zd,
        "r_missing": z0 - zr,
        "labels": labels,
        "base_clean_logits": base_clean_logits,
        "base_degraded_logits": base_degraded_logits,
        "adapter_logits": adapter_logits,
    }


def probabilities(logits: torch.Tensor) -> torch.Tensor:
    return logits.softmax(dim=-1)[..., 1].mean(dim=-1)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end + 1) / 2
        start = end
    return ranks


def correlation(x: np.ndarray, y: np.ndarray) -> dict:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return {"pearson": None, "spearman": None}
    return {
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "spearman": float(np.corrcoef(_rank(x), _rank(y))[0, 1]),
    }


def transition_summary(
    base_probability: torch.Tensor,
    new_probability: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
) -> dict:
    base_correct = (base_probability >= threshold) == labels.bool()
    new_correct = (new_probability >= threshold) == labels.bool()
    wrong_to_correct = ~base_correct & new_correct
    correct_to_wrong = base_correct & ~new_correct
    total = len(labels)
    return {
        "wrong_to_correct": int(wrong_to_correct.sum()),
        "correct_to_wrong": int(correct_to_wrong.sum()),
        "net_correction_gain": float(
            (wrong_to_correct.sum() - correct_to_wrong.sum()) / total
        ),
        "wrong_to_correct_rate": float(wrong_to_correct.sum() / total),
        "correct_to_wrong_rate": float(correct_to_wrong.sum() / total),
    }


def evaluate_state(state: dict, threshold: float) -> dict:
    labels = state["labels"]
    base_clean_probability = probabilities(state["base_clean_logits"])
    base_degraded_probability = probabilities(state["base_degraded_logits"])
    adapter_probability = probabilities(state["adapter_logits"])
    base_clean_correct = (base_clean_probability >= threshold) == labels.bool()
    base_degraded_correct = (
        (base_degraded_probability >= threshold) == labels.bool()
    )
    adapter_correct = (adapter_probability >= threshold) == labels.bool()
    return {
        "base_clean": binary_metrics(
            labels.cpu().numpy(), base_clean_probability.cpu().numpy(), threshold
        ),
        "base_degraded": binary_metrics(
            labels.cpu().numpy(), base_degraded_probability.cpu().numpy(), threshold
        ),
        "adapter": binary_metrics(
            labels.cpu().numpy(), adapter_probability.cpu().numpy(), threshold
        ),
        "transition": transition_summary(
            base_degraded_probability,
            adapter_probability,
            labels,
            threshold,
        ),
        "margins": {
            "base_clean": float(
                correct_class_margin(state["base_clean_logits"], labels).mean()
            ),
            "base_degraded": float(
                correct_class_margin(
                    state["base_degraded_logits"], labels
                ).mean()
            ),
            "adapter": float(
                correct_class_margin(state["adapter_logits"], labels).mean()
            ),
        },
        "flags": {
            "clean_to_degraded_error": base_clean_correct & ~base_degraded_correct,
            "recovered_by_adapter": base_clean_correct
            & ~base_degraded_correct
            & adapter_correct,
            "still_wrong_after_adapter": base_clean_correct
            & ~base_degraded_correct
            & ~adapter_correct,
            "adapter_harm": base_degraded_correct & ~adapter_correct,
        },
        "probabilities": {
            "base_clean": base_clean_probability,
            "base_degraded": base_degraded_probability,
            "adapter": adapter_probability,
        },
    }


def distance_summary(state: dict, evaluation: dict, video_ids: np.ndarray) -> tuple[dict, list[dict]]:
    z0, zd, zr = state["z0"], state["zd"], state["zr"]
    labels = state["labels"]
    cos_0d = torch.nn.functional.cosine_similarity(z0, zd, dim=-1).mean(-1)
    cos_0r = torch.nn.functional.cosine_similarity(z0, zr, dim=-1).mean(-1)
    cos_dr = torch.nn.functional.cosine_similarity(zd, zr, dim=-1).mean(-1)
    norm_deg = state["r_degradation"].norm(dim=-1).mean(-1)
    norm_rec = state["r_recovered"].norm(dim=-1).mean(-1)
    norm_miss = state["r_missing"].norm(dim=-1).mean(-1)
    margin_change = (
        correct_class_margin(state["adapter_logits"], labels)
        - correct_class_margin(state["base_degraded_logits"], labels)
    )
    full_mask = torch.ones_like(cos_0d, dtype=torch.bool)
    groups = {
        "all": full_mask,
        "real": ~labels.bool(),
        "fake": labels.bool(),
        **evaluation["flags"],
    }
    summary = {}
    for name, group in groups.items():
        if not bool(group.any()):
            summary[name] = {"count": 0}
            continue
        summary[name] = {
            "count": int(group.sum()),
            "cosine_original_degraded": float(cos_0d[group].mean()),
            "cosine_original_restored": float(cos_0r[group].mean()),
            "cosine_degraded_restored": float(cos_dr[group].mean()),
            "degradation_residual_norm": float(norm_deg[group].mean()),
            "recovered_residual_norm": float(norm_rec[group].mean()),
            "missing_residual_norm": float(norm_miss[group].mean()),
            "margin_recovery": float(margin_change[group].mean()),
        }
    correlations = {
        "cosine_gain_vs_margin_recovery": correlation(
            (cos_0r - cos_0d).cpu().numpy(),
            margin_change.cpu().numpy(),
        ),
        "missing_norm_vs_abs_margin_recovery": correlation(
            norm_miss.cpu().numpy(),
            margin_change.abs().cpu().numpy(),
        ),
    }
    rows = []
    for index, video_id in enumerate(video_ids):
        rows.append(
            {
                "video_id": str(video_id),
                "label": "fake" if bool(labels[index]) else "real",
                "cosine_original_degraded": float(cos_0d[index]),
                "cosine_original_restored": float(cos_0r[index]),
                "cosine_degraded_restored": float(cos_dr[index]),
                "degradation_residual_norm": float(norm_deg[index]),
                "recovered_residual_norm": float(norm_rec[index]),
                "missing_residual_norm": float(norm_miss[index]),
                "margin_recovery": float(margin_change[index]),
                "adapter_delta_probability": float(
                    evaluation["probabilities"]["adapter"][index]
                    - evaluation["probabilities"]["base_degraded"][index]
                ),
                **{
                    name: bool(group[index])
                    for name, group in evaluation["flags"].items()
                },
            }
        )
    return {"groups": summary, "correlations": correlations}, rows


@torch.no_grad()
def train_importance(
    model: GenDAdapter,
    state: dict,
) -> dict[str, torch.Tensor]:
    zr = state["zr"]
    labels = state["labels"]
    missing = state["r_missing"]
    weight_fake_minus_real = (
        model.head.weight[1] - model.head.weight[0]
    ).detach()
    sign = torch.where(labels.bool(), 1.0, -1.0)
    gradient = sign[:, None, None] * weight_fake_minus_real[None, None, :]
    effective = gradient - (
        gradient * zr
    ).sum(dim=-1, keepdim=True) * zr
    directional = effective * missing
    return {
        "magnitude": missing.abs().mean(dim=(0, 1)),
        "classifier": effective.abs().mean(dim=(0, 1)),
        "discriminative": torch.relu(directional).mean(dim=(0, 1)),
        "signed_discriminative": directional.mean(dim=(0, 1)),
        "mean_square_residual": missing.square().mean(dim=(0, 1)),
    }


def select_mask(
    selector: str,
    scores: dict[str, torch.Tensor],
    ratio: float,
    seed: int,
) -> torch.Tensor:
    dimension = len(scores["magnitude"])
    count = max(1, min(dimension, int(round(dimension * ratio))))
    if selector == "random":
        residual_energy = scores["mean_square_residual"]
        generator = torch.Generator(device=residual_energy.device)
        generator.manual_seed(seed + count)
        candidates = torch.argsort(
            torch.rand(
                2000,
                dimension,
                device=residual_energy.device,
                generator=generator,
            ),
            dim=1,
        )[:, :count]
        target = None
        candidate_energy = residual_energy[candidates].sum(dim=1)
        # Match the strongest candidate selector so random controls share
        # approximately the same residual energy.
        target = torch.topk(residual_energy, count).values.sum()
        index = int((candidate_energy - target).abs().argmin())
        mask = torch.zeros(dimension, dtype=torch.bool, device=residual_energy.device)
        mask[candidates[index]] = True
        return mask
    score = scores[selector]
    return torch.topk(score, count).indices.sort().values


def mask_tensor(indices: torch.Tensor, dimension: int) -> torch.Tensor:
    mask = torch.zeros(dimension, dtype=torch.bool, device=indices.device)
    mask[indices] = True
    return mask


@torch.inference_mode()
def intervene(
    model: GenDAdapter,
    state: dict,
    mask: torch.Tensor,
    direction: str,
) -> dict:
    vector = mask.to(state["zr"].device)
    if direction == "add":
        start = state["zr"]
        features = start + vector * state["r_missing"]
    elif direction == "reverse":
        # r_missing is defined in Adapter output space. Its clean endpoint is
        # therefore A(z_original), not the pre-Adapter GenD feature z_original.
        start = state["zc"]
        features = start - vector * state["r_missing"]
    else:
        raise ValueError(direction)
    features = torch.nn.functional.normalize(features, p=2, dim=-1)
    logits = model.head(features)
    probability = probabilities(logits)
    base_logits = model.head(start)
    base_probability = probabilities(base_logits)
    margin = correct_class_margin(logits, state["labels"])
    base_margin = correct_class_margin(base_logits, state["labels"])
    return {
        "metrics": binary_metrics(
            state["labels"].cpu().numpy(),
            probability.cpu().numpy(),
            0.5,
        ),
        "margin": float(margin.mean()),
        "margin_change": float((margin - base_margin).mean()),
        **transition_summary(
            base_probability,
            probability,
            state["labels"],
            0.5,
        ),
        "probability": probability,
        "logits": logits,
        "base_probability": base_probability,
        "base_logits": base_logits,
        "base_margin": float(base_margin.mean()),
    }


def selection_and_interventions(
    model: GenDAdapter,
    train_state: dict,
    validation_state: dict,
    test_state: dict,
    ratios: tuple[float, ...],
    threshold: float,
    seed: int,
) -> tuple[dict, list[dict], np.ndarray]:
    scores = train_importance(model, train_state)
    dimension = len(scores["magnitude"])
    validation_rows = []
    masks_by_selector_ratio = {}
    for selector in SELECTORS:
        for ratio in ratios:
            indices = select_mask(selector, scores, ratio, seed)
            mask = mask_tensor(indices, dimension)
            masks_by_selector_ratio[(selector, ratio)] = mask
            add = intervene(model, validation_state, mask, "add")
            reverse = intervene(model, validation_state, mask, "reverse")
            validation_rows.append(
                {
                    "selector": selector,
                    "ratio": ratio,
                    "selected_dimensions": int(mask.sum()),
                    "add_accuracy": add["metrics"]["accuracy"],
                    "add_balanced_accuracy": add["metrics"]["balanced_accuracy"],
                    "add_auc": add["metrics"]["auc"],
                    "add_margin": add["margin"],
                    "add_margin_change": add["margin_change"],
                    "reverse_accuracy": reverse["metrics"]["accuracy"],
                    "reverse_balanced_accuracy": reverse["metrics"][
                        "balanced_accuracy"
                    ],
                    "reverse_margin": reverse["margin"],
                    "reverse_margin_change": reverse["margin_change"],
                }
            )

    selected = {}
    for selector in SELECTORS:
        candidates = [
            row for row in validation_rows if row["selector"] == selector
        ]
        best = max(
            candidates,
            key=lambda row: (
                row["add_balanced_accuracy"] or 0.0,
                row["add_margin"],
            ),
        )
        selected[selector] = best

    test_rows = []
    test_probabilities = {}
    for selector, choice in selected.items():
        mask = masks_by_selector_ratio[(selector, choice["ratio"])]
        add = intervene(model, test_state, mask, "add")
        reverse = intervene(model, test_state, mask, "reverse")
        test_probabilities[f"{selector}_add"] = add["probability"]
        test_probabilities[f"{selector}_reverse"] = reverse["probability"]
        test_rows.append(
            {
                "selector": selector,
                "ratio": choice["ratio"],
                "selected_dimensions": int(mask.sum()),
                "add_accuracy": add["metrics"]["accuracy"],
                "add_balanced_accuracy": add["metrics"]["balanced_accuracy"],
                "add_auc": add["metrics"]["auc"],
                "add_margin": add["margin"],
                "add_margin_change": add["margin_change"],
                "add_wrong_to_correct": add["wrong_to_correct"],
                "add_correct_to_wrong": add["correct_to_wrong"],
                "add_net_correction_gain": add["net_correction_gain"],
                "reverse_accuracy": reverse["metrics"]["accuracy"],
                "reverse_balanced_accuracy": reverse["metrics"][
                    "balanced_accuracy"
                ],
                "reverse_auc": reverse["metrics"]["auc"],
                "reverse_margin": reverse["margin"],
                "reverse_margin_change": reverse["margin_change"],
                "reverse_wrong_to_correct": reverse["wrong_to_correct"],
                "reverse_correct_to_wrong": reverse["correct_to_wrong"],
                "reverse_net_correction_gain": reverse["net_correction_gain"],
            }
        )

    full_mask = torch.ones(dimension, dtype=torch.bool, device=model.head.weight.device)
    oracle = intervene(model, test_state, full_mask, "add")
    test_probabilities["oracle_add"] = oracle["probability"]
    return {
        "importance_scores": scores,
        "validation_rows": validation_rows,
        "selected": selected,
        "test_rows": test_rows,
        "test_probabilities": test_probabilities,
        "oracle": oracle,
    }, validation_rows, np.asarray(
        [int(mask.sum()) for mask in masks_by_selector_ratio.values()]
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def plot_diagnostics(
    output_dir: Path,
    per_condition: dict,
    distance_rows: dict[str, list[dict]],
    test_predictions: dict[str, dict[str, np.ndarray]],
    test_labels: dict[str, np.ndarray],
    importance: dict[str, torch.Tensor],
    validation_rows: dict[str, list[dict]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    condition_names = list(per_condition)
    colors = ["#1F4E79", "#2A9D8F", "#E76F51"]

    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=300)
    x = np.arange(len(condition_names))
    width = 0.25
    raw = [
        per_condition[name]["distances"]["test"]["groups"]["all"][
            "cosine_original_degraded"
        ]
        for name in condition_names
    ]
    restored = [
        per_condition[name]["distances"]["test"]["groups"]["all"][
            "cosine_original_restored"
        ]
        for name in condition_names
    ]
    missing_norm = [
        per_condition[name]["distances"]["test"]["groups"]["all"][
            "missing_residual_norm"
        ]
        for name in condition_names
    ]
    ax.bar(x - width, raw, width, label="Clean-degraded cosine", color=colors[0])
    ax.bar(x, restored, width, label="Clean-restored cosine", color=colors[1])
    ax2 = ax.twinx()
    ax2.plot(
        x + width,
        missing_norm,
        marker="o",
        color=colors[2],
        label="Missing-residual norm",
    )
    ax.set_xticks(x, [CONDITIONS[name] for name in condition_names])
    ax.set_ylabel("Cosine similarity")
    ax2.set_ylabel("Residual L2 norm")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        lines + lines2,
        labels + labels2,
        frameon=False,
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.17),
        ncol=3,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "residual_distance_comparison.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=300)
    data = [
        [
            row["missing_residual_norm"]
            for row in distance_rows[name]
        ]
        for name in condition_names
    ]
    ax.boxplot(data, tick_labels=[CONDITIONS[name] for name in condition_names])
    ax.set_ylabel("Missing-residual norm")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(output_dir / "missing_residual_distribution.png", bbox_inches="tight")
    plt.close(fig)

    primary = "blur_k9"
    mag = importance[primary]["magnitude"].cpu().numpy()
    disc = importance[primary]["discriminative"].cpu().numpy()
    cls = importance[primary]["classifier"].cpu().numpy()
    fig, ax = plt.subplots(figsize=(6.2, 3.2), dpi=300)
    ax.scatter(mag, disc, s=10, alpha=0.55, color=colors[0], label="Discriminative")
    ax.scatter(mag, cls, s=7, alpha=0.25, color=colors[1], label="Classifier sensitivity")
    ax.set_xlabel("Mean missing-residual magnitude")
    ax.set_ylabel("Importance score")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(color="#E5E7EB", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(output_dir / "importance_vs_magnitude.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.9), dpi=300, sharey=True)
    for axis, condition in zip(axes, condition_names):
        rows = validation_rows[condition]
        for selector in SELECTORS:
            selected = [row for row in rows if row["selector"] == selector]
            axis.plot(
                [row["ratio"] * 100 for row in selected],
                [100 * row["add_balanced_accuracy"] for row in selected],
                marker="o",
                markersize=3,
                label=selector,
            )
        axis.set_title(CONDITIONS[condition], fontsize=8)
        axis.set_xlabel("Add-back ratio (%)")
        axis.grid(color="#E5E7EB", linewidth=0.6)
    axes[0].set_ylabel("Balanced accuracy (%)")
    axes[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(output_dir / "addback_ratio_curves.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.2), dpi=300)
    for col, condition in enumerate(condition_names):
        rows = per_condition[condition]["interventions"]["test_rows"]
        selector_names = [row["selector"] for row in rows]
        positions = np.arange(len(rows))
        axes[0, col].bar(
            positions,
            [100 * row["add_balanced_accuracy"] for row in rows],
            color=colors[0],
        )
        axes[0, col].set_xticks(
            positions,
            selector_names,
            rotation=25,
            ha="right",
            fontsize=7,
        )
        axes[0, col].set_title(CONDITIONS[condition], fontsize=8)
        axes[0, col].set_ylabel("Add-back balanced acc. (%)")
        axes[0, col].grid(color="#E5E7EB", linewidth=0.6)
        axes[1, col].bar(
            positions,
            [100 * row["reverse_net_correction_gain"] for row in rows],
            color=colors[2],
        )
        axes[1, col].set_xticks(
            positions,
            selector_names,
            rotation=25,
            ha="right",
            fontsize=7,
        )
        axes[1, col].axhline(0, color="#6B7280", linewidth=0.8)
        axes[1, col].set_ylabel("Reverse net correction (pp)")
        axes[1, col].grid(color="#E5E7EB", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output_dir / "forward_reverse_interventions.png", bbox_inches="tight")
    plt.close(fig)

    labels = ["Real", "Fake"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), dpi=300)
    for axis, label_name in zip(axes, labels):
        rows = np.asarray(
            [
                row["margin_recovery"]
                for row in distance_rows[primary]
                if row["label"].lower() == label_name.lower()
            ]
        )
        axis.hist(rows, bins=18, color=colors[0], alpha=0.8)
        axis.set_title(label_name)
        axis.set_xlabel("Adapter margin recovery")
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output_dir / "real_fake_margin_recovery.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 3.2), dpi=300)
    labels_np = test_labels[primary]
    predictions = test_predictions[primary]
    for key, label, color in (
        ("base", "GenD", colors[0]),
        ("adapter", "Adapter", colors[1]),
        ("discriminative_add", "Discriminative add-back", colors[2]),
    ):
        prob = predictions[key]
        thresholds = np.unique(np.r_[0, prob, 1])
        tpr, fpr = [], []
        for threshold in thresholds:
            pred = prob >= threshold
            tpr.append(np.mean(pred[labels_np.astype(bool)]))
            fpr.append(np.mean(pred[~labels_np.astype(bool)]))
        ax.plot(fpr, tpr, label=label, color=color)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(color="#E5E7EB", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(output_dir / "roc_comparison_blur.png", bbox_inches="tight")
    plt.close(fig)


def analyze_condition(
    condition: str,
    data: dict,
    checkpoint: dict,
    model: GenDAdapter,
    ratios: tuple[float, ...],
    threshold: float,
    seed: int,
    device: torch.device,
) -> dict:
    outputs = {}
    indices = {
        split: split_indices(checkpoint, split, data["video_ids"])
        for split in ("train", "validation", "test")
    }
    states = {
        split: feature_state(model, data, index, device)
        for split, index in indices.items()
    }
    evaluations = {
        split: evaluate_state(state, threshold)
        for split, state in states.items()
    }
    distance_results = {}
    distance_rows = {}
    for split in ("train", "validation", "test"):
        summary, rows = distance_summary(
            states[split],
            evaluations[split],
            data["video_ids"][indices[split]],
        )
        distance_results[split] = summary
        distance_rows[split] = rows
    interventions, validation_rows, _ = selection_and_interventions(
        model,
        states["train"],
        states["validation"],
        states["test"],
        ratios,
        threshold,
        seed,
    )
    return {
        "indices": indices,
        "states": states,
        "evaluations": evaluations,
        "distances": distance_results,
        "distance_rows": distance_rows,
        "interventions": interventions,
        "validation_rows": validation_rows,
    }


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "blur_k9": args.blur_features,
        "jpeg_q30": args.jpeg_features,
        "resize_25": args.resize_features,
    }
    caches = {name: load_npz(path) for name, path in paths.items()}
    model, checkpoint = load_model(args.checkpoint, device)
    per_condition = {}
    all_interventions = []
    all_importance = {}
    validation_by_condition = {}
    distance_rows_by_condition = {}
    test_predictions = {}
    test_labels = {}
    condition_summary = {}
    for condition, data in caches.items():
        result = analyze_condition(
            condition,
            data,
            checkpoint,
            model,
            RATIOS,
            args.threshold,
            args.seed,
            device,
        )
        condition_summary[condition] = {
            "train": result["evaluations"]["train"],
            "validation": result["evaluations"]["validation"],
            "test": result["evaluations"]["test"],
            "distances": result["distances"],
            "interventions": {
                "selected": result["interventions"]["selected"],
                "test_rows": result["interventions"]["test_rows"],
                "oracle": {
                    "metrics": result["interventions"]["oracle"]["metrics"],
                    "margin": result["interventions"]["oracle"]["margin"],
                    "margin_change": result["interventions"]["oracle"]["margin_change"],
                },
            },
        }
        all_importance[condition] = result["interventions"]["importance_scores"]
        validation_by_condition[condition] = result["interventions"][
            "validation_rows"
        ]
        distance_rows_by_condition[condition] = result["distance_rows"]["test"]
        test_labels[condition] = result["states"]["test"]["labels"].cpu().numpy()
        test_predictions[condition] = {
            "base": result["evaluations"]["test"]["probabilities"][
                "base_degraded"
            ]
            .cpu()
            .numpy(),
            "adapter": result["evaluations"]["test"]["probabilities"]["adapter"]
            .cpu()
            .numpy(),
            **{
                key: probability.cpu().numpy()
                for key, probability in result["interventions"][
                    "test_probabilities"
                ].items()
            },
        }
        # Save the full per-condition diagnostic without tensors.
        result_dir = output_dir / condition
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "diagnostic.json").write_text(
            json.dumps(
                jsonable({
                    "condition": condition,
                    "model_scale": float(
                        model.adapter.scale.detach().cpu()
                    ),
                    "evaluation": {
                        split: {
                            key: value
                            for key, value in evaluation.items()
                            if key not in {"flags", "probabilities"}
                        }
                        for split, evaluation in result["evaluations"].items()
                    },
                    "distance_groups": {
                        split: distance["groups"]
                        for split, distance in result["distances"].items()
                    },
                    "distance_correlations": {
                        split: distance["correlations"]
                        for split, distance in result["distances"].items()
                    },
                    "validation_interventions": result["validation_rows"],
                    "selected_test_interventions": result["interventions"]["test_rows"],
                    "selected": result["interventions"]["selected"],
                    "oracle": {
                        "metrics": result["interventions"]["oracle"]["metrics"],
                        "margin": result["interventions"]["oracle"]["margin"],
                        "margin_change": result["interventions"]["oracle"][
                            "margin_change"
                        ],
                    },
                }),
                indent=2,
            ),
            encoding="utf-8",
        )
        write_csv(
            result_dir / "test_video_features.csv",
            distance_rows_by_condition[condition],
        )
        np.savez_compressed(
            result_dir / "importance_scores.npz",
            **{
                key: value.detach().cpu().numpy()
                for key, value in all_importance[condition].items()
            },
        )
        for row in result["interventions"]["test_rows"]:
            all_interventions.append({"condition": condition, **row})
    # Compact summaries used by the report and later visualization.
    (output_dir / "summary.json").write_text(
        json.dumps(
            jsonable({
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "model_scale": float(model.adapter.scale.detach().cpu()),
                "threshold": args.threshold,
                "ratios": list(RATIOS),
                "conditions": condition_summary,
            }),
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(output_dir / "interventions_test.csv", all_interventions)
    write_csv(
        output_dir / "interventions_validation.csv",
        [
            {"condition": condition, **row}
            for condition, rows in validation_by_condition.items()
            for row in rows
        ],
    )
    plot_diagnostics(
        output_dir,
        condition_summary,
        distance_rows_by_condition,
        test_predictions,
        test_labels,
        all_importance,
        validation_by_condition,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
