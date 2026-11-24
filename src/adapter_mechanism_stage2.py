from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from .gend_model import choose_device
from .metrics import binary_metrics
from .residual_diagnostics import load_npz, split_indices
from .residual_stage2 import (
    classification_report,
    head_original,
    load_model,
    probabilities,
)


RESIDUAL_NAMES = ("R1", "R2", "R3", "R4", "R5", "R6")
ERROR_GROUPS = {
    "A_successful_correction": (True, False, True),
    "B_persistent_degradation_error": (True, False, False),
    "C_adapter_introduced_error": (True, True, False),
    "D_persistent_original_error": (False, False, False),
    "E_stable_correct": (True, True, True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Uniform vs restoration Adapter mechanism diagnostics."
    )
    parser.add_argument("--blur-features", required=True)
    parser.add_argument("--jpeg-features", required=True)
    parser.add_argument("--resize-features", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--independent-c", type=float, default=0.1)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metric_row(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
) -> dict:
    report = classification_report(logits, labels, threshold)
    return {
        "accuracy": report["accuracy"],
        "balanced_accuracy": report["balanced_accuracy"],
        "auc": report["auc"],
        "f1": report["f1"],
        "false_positive_rate": report["false_positive_rate"],
        "false_negative_rate": report["false_negative_rate"],
        "cross_entropy": report["cross_entropy"],
        "correct_probability": report["correct_probability"],
        "mean_frame_logit_margin": report["mean_frame_logit_margin"],
        "mean_confidence": report["mean_confidence"],
        "mean_abs_logit": report["mean_abs_logit"],
        "frame_video_rule_disagreement": report[
            "frame_video_rule_disagreement"
        ],
    }


@torch.inference_mode()
def build_states(
    uniform,
    restoration,
    clean,
    data: dict,
    indices: np.ndarray,
    device: torch.device,
) -> dict:
    z0 = torch.from_numpy(data["features_original"][indices]).to(device).float()
    zd = torch.from_numpy(data["features_blur"][indices]).to(device).float()
    zu = uniform.adapter(zd)
    zf = restoration.adapter(zd)
    zuc = uniform.adapter(z0)
    zfc = restoration.adapter(z0)
    zclean = clean.adapter(zd)
    labels = torch.from_numpy(
        data["label_index"][indices].astype(np.int64)
    ).to(device)
    return {
        "z0": z0,
        "zd": zd,
        "zu": zu,
        "zf": zf,
        "zuc": zuc,
        "zfc": zfc,
        "zclean": zclean,
        "R1": z0 - zd,
        "R2": z0 - zu,
        "R3": zuc - zu,
        "R4": z0 - zf,
        "R5": zfc - zf,
        "R6": zu - zf,
        "labels": labels,
        "group_ids": data["group_ids"][indices],
        "video_ids": data["video_ids"][indices],
    }


def direction(head: torch.nn.Linear) -> torch.Tensor:
    weight = head.weight[1] - head.weight[0]
    return weight / weight.norm().clamp_min(1e-8)


def endpoint_pair(residual_name: str) -> tuple[str, str]:
    return {
        "R1": ("z0", "zd"),
        "R2": ("z0", "zu"),
        "R3": ("zuc", "zu"),
        "R4": ("z0", "zf"),
        "R5": ("zfc", "zf"),
        "R6": ("zu", "zf"),
    }[residual_name]


@torch.inference_mode()
def residual_rows(
    states: dict,
    seed: int,
    split: str,
    uniform_head: torch.nn.Linear,
    restoration_head: torch.nn.Linear,
    original_head: torch.nn.Linear,
) -> list[dict]:
    labels = states["labels"]
    rows = []
    for residual_name in RESIDUAL_NAMES:
        residual = states[residual_name]
        clean_endpoint, degraded_endpoint = endpoint_pair(residual_name)
        endpoint_cosine = torch.nn.functional.cosine_similarity(
            states[clean_endpoint], states[degraded_endpoint], dim=-1
        ).mean(dim=-1)
        norm = residual.norm(dim=-1).mean(dim=-1)
        for head_name, head in (
            ("original", original_head),
            ("uniform", uniform_head),
            ("restoration", restoration_head),
        ):
            classifier_direction = direction(head)
            signed_direction = torch.where(
                labels.bool(), 1.0, -1.0
            )[:, None, None] * classifier_direction
            projection = (residual * signed_direction).sum(dim=-1)
            parallel = (
                projection[..., None] * signed_direction
            )
            parallel_fraction = (
                parallel.square().sum(dim=-1)
                / residual.square().sum(dim=-1).clamp_min(1e-12)
            )
            for group_name, mask in (
                ("all", torch.ones_like(labels, dtype=torch.bool)),
                ("real", ~labels.bool()),
                ("fake", labels.bool()),
            ):
                if not bool(mask.any()):
                    continue
                rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "residual": residual_name,
                        "classifier": head_name,
                        "group": group_name,
                        "videos": int(mask.sum()),
                        "residual_norm_mean": float(norm[mask].mean()),
                        "residual_norm_std": float(norm[mask].std(unbiased=False)),
                        "endpoint_cosine_mean": float(
                            endpoint_cosine[mask].mean()
                        ),
                        "classifier_projection_mean": float(
                            projection[mask].mean()
                        ),
                        "classifier_projection_std": float(
                            projection[mask].std(unbiased=False)
                        ),
                        "parallel_energy_fraction_mean": float(
                            parallel_fraction[mask].mean()
                        ),
                    }
                )
    return rows


def transition_group(
    clean_correct: bool,
    degraded_correct: bool,
    adapter_correct: bool,
) -> str:
    state = (clean_correct, degraded_correct, adapter_correct)
    for name, pattern in ERROR_GROUPS.items():
        if state == pattern:
            return name
    return "other"


def transition_code(
    clean_correct: bool,
    degraded_correct: bool,
    adapter_correct: bool,
) -> str:
    return "".join(
        "1" if value else "0"
        for value in (clean_correct, degraded_correct, adapter_correct)
    )


@torch.inference_mode()
def error_conditioned_rows(
    states: dict,
    seed: int,
    split: str,
    original_head: torch.nn.Linear,
    uniform_head: torch.nn.Linear,
    threshold: float,
) -> tuple[list[dict], list[dict]]:
    labels = states["labels"]
    clean_logits = original_head(states["z0"])
    degraded_logits = original_head(states["zd"])
    adapter_logits = uniform_head(states["zu"])
    clean_probability = probabilities(clean_logits)
    degraded_probability = probabilities(degraded_logits)
    adapter_probability = probabilities(adapter_logits)
    clean_correct = (clean_probability >= threshold) == labels.bool()
    degraded_correct = (degraded_probability >= threshold) == labels.bool()
    adapter_correct = (adapter_probability >= threshold) == labels.bool()
    group_names = [
        transition_group(
            bool(clean_correct[index]),
            bool(degraded_correct[index]),
            bool(adapter_correct[index]),
        )
        for index in range(len(labels))
    ]
    codes = [
        transition_code(
            bool(clean_correct[index]),
            bool(degraded_correct[index]),
            bool(adapter_correct[index]),
        )
        for index in range(len(labels))
    ]
    # Full R2/R3 add-back are Oracle diagnostics because they use clean refs.
    r2_features = torch.nn.functional.normalize(
        states["zu"] + states["R2"], dim=-1
    )
    r3_features = torch.nn.functional.normalize(
        states["zu"] + states["R3"], dim=-1
    )
    r2_correct = (
        probabilities(uniform_head(r2_features)) >= threshold
    ) == labels.bool()
    r3_correct = (
        probabilities(uniform_head(r3_features)) >= threshold
    ) == labels.bool()
    clean_correct_probability = torch.where(
        labels.bool(), clean_probability, 1 - clean_probability
    )
    degraded_correct_probability = torch.where(
        labels.bool(), degraded_probability, 1 - degraded_probability
    )
    adapter_correct_probability = torch.where(
        labels.bool(), adapter_probability, 1 - adapter_probability
    )
    clean_margin = (
        clean_logits.gather(
            -1, labels[:, None, None].expand(-1, clean_logits.shape[1], 1)
        ).squeeze(-1)
        - clean_logits.gather(
            -1,
            (1 - labels)[:, None, None].expand(-1, clean_logits.shape[1], 1),
        ).squeeze(-1)
    ).mean(-1)
    degraded_margin = (
        degraded_logits.gather(
            -1, labels[:, None, None].expand(-1, degraded_logits.shape[1], 1)
        ).squeeze(-1)
        - degraded_logits.gather(
            -1,
            (1 - labels)[:, None, None].expand(
                -1, degraded_logits.shape[1], 1
            ),
        ).squeeze(-1)
    ).mean(-1)
    adapter_margin = (
        adapter_logits.gather(
            -1, labels[:, None, None].expand(-1, adapter_logits.shape[1], 1)
        ).squeeze(-1)
        - adapter_logits.gather(
            -1,
            (1 - labels)[:, None, None].expand(-1, adapter_logits.shape[1], 1),
        ).squeeze(-1)
    ).mean(-1)
    rows = []
    transition_rows = []
    all_groups = (*ERROR_GROUPS, "other")
    for group_name in all_groups:
        if group_name == "other":
            mask_indices = [
                index
                for index, value in enumerate(group_names)
                if value == "other"
            ]
        else:
            mask_indices = [
                index
                for index, value in enumerate(group_names)
                if value == group_name
            ]
        if not mask_indices:
            continue
        mask = torch.zeros_like(labels, dtype=torch.bool)
        mask[mask_indices] = True
        row = {
            "seed": seed,
            "split": split,
            "group": group_name,
            "videos": int(mask.sum()),
            "real_videos": int((~labels.bool() & mask).sum()),
            "fake_videos": int((labels.bool() & mask).sum()),
            "clean_correct_probability": float(
                clean_correct_probability[mask].mean()
            ),
            "degraded_correct_probability": float(
                degraded_correct_probability[mask].mean()
            ),
            "adapter_correct_probability": float(
                adapter_correct_probability[mask].mean()
            ),
            "clean_margin": float(clean_margin[mask].mean()),
            "degraded_margin": float(degraded_margin[mask].mean()),
            "adapter_margin": float(adapter_margin[mask].mean()),
            "R2_full_add_accuracy": float(r2_correct[mask].float().mean()),
            "R3_full_add_accuracy": float(r3_correct[mask].float().mean()),
            "R2_full_add_wrong_to_correct": int(
                ((~adapter_correct) & r2_correct & mask).sum()
            ),
            "R2_full_add_correct_to_wrong": int(
                (adapter_correct & ~r2_correct & mask).sum()
            ),
            "R3_full_add_wrong_to_correct": int(
                ((~adapter_correct) & r3_correct & mask).sum()
            ),
            "R3_full_add_correct_to_wrong": int(
                (adapter_correct & ~r3_correct & mask).sum()
            ),
        }
        for residual_name in ("R1", "R2", "R3", "R4", "R5", "R6"):
            row[f"{residual_name}_norm"] = float(
                states[residual_name].norm(dim=-1).mean(dim=-1)[mask].mean()
            )
        row["cosine_clean_degraded"] = float(
            torch.nn.functional.cosine_similarity(
                states["z0"], states["zd"], dim=-1
            ).mean(-1)[mask].mean()
        )
        row["cosine_clean_uniform"] = float(
            torch.nn.functional.cosine_similarity(
                states["z0"], states["zu"], dim=-1
            ).mean(-1)[mask].mean()
        )
        row["cosine_clean_restoration"] = float(
            torch.nn.functional.cosine_similarity(
                states["z0"], states["zf"], dim=-1
            ).mean(-1)[mask].mean()
        )
        rows.append(row)
    code_counts = defaultdict(lambda: {"videos": 0, "real": 0, "fake": 0})
    for code, group_name in zip(codes, group_names):
        code_counts[(code, group_name)]["videos"] += 1
    for index, (code, group_name) in enumerate(zip(codes, group_names)):
        code_counts[(code, group_name)][
            "fake" if bool(labels[index]) else "real"
        ] += 1
    for (code, group_name), counts in sorted(code_counts.items()):
        transition_rows.append(
            {
                "seed": seed,
                "split": split,
                "transition_code": code,
                "group": group_name,
                **counts,
            }
        )
    return rows, transition_rows


def adapter_comparison_rows(
    states: dict,
    seed: int,
    condition: str,
    original_head: torch.nn.Linear,
    uniform,
    restoration,
    clean,
    threshold: float,
) -> list[dict]:
    labels = states["labels"]
    return [
        {
            "seed": seed,
            "condition": condition,
            "split": "test",
            "method": method,
            **metric_row(logits, labels, threshold),
        }
        for method, logits in (
            ("GenD", original_head(states["zd"])),
            ("Uniform", uniform.head(states["zu"])),
            ("Restoration", restoration.head(states["zf"])),
            ("Clean-only", clean.head(states["zclean"])),
        )
    ]


def shared_head_rows(
    states: dict,
    seed: int,
    condition: str,
    split: str,
    heads: dict[str, torch.nn.Linear],
    threshold: float,
) -> list[dict]:
    rows = []
    feature_names = ("z0", "zd", "zu", "zf", "zuc", "zfc")
    for head_name, head in heads.items():
        for feature_name in feature_names:
            rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "split": split,
                    "head": head_name,
                    "features": feature_name,
                    **metric_row(
                        head(states[feature_name]),
                        states["labels"],
                        threshold,
                    ),
                }
            )
    return rows


def flatten(values: torch.Tensor) -> np.ndarray:
    return values.detach().cpu().numpy().reshape(-1, values.shape[-1])


def independent_head_rows(
    train_states: dict,
    validation_states: dict,
    test_states: dict,
    seed: int,
    condition: str,
    c_value: float,
    threshold: float,
) -> list[dict]:
    rows = []
    train_labels = np.repeat(
        train_states["labels"].cpu().numpy(), train_states["z0"].shape[1]
    )
    for feature_name in ("z0", "zd", "zu", "zf"):
        model = LogisticRegression(
            C=c_value,
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        )
        model.fit(flatten(train_states[feature_name]), train_labels)
        for split, states in (
            ("validation", validation_states),
            ("test", test_states),
        ):
            frame_probability = model.predict_proba(
                flatten(states[feature_name])
            )[:, 1]
            video_probability = frame_probability.reshape(
                states["z0"].shape[0], states["z0"].shape[1]
            ).mean(axis=1)
            labels = states["labels"].cpu().numpy()
            metrics = binary_metrics(labels, video_probability, threshold)
            rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "split": split,
                    "features": feature_name,
                    "C": c_value,
                    "coefficient_norm": float(
                        np.linalg.norm(model.coef_[0])
                    ),
                    **metrics,
                }
            )
    return rows


def aggregate(rows: list[dict], group_keys: tuple[str, ...]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    output = []
    numeric_keys = [
        key
        for key, value in rows[0].items()
        if key not in group_keys
        and key != "seed"
        and isinstance(value, (int, float, np.integer, np.floating))
    ]
    for key, values in sorted(grouped.items()):
        result = dict(zip(group_keys, key))
        result["seeds"] = len({row.get("seed") for row in values})
        for metric in numeric_keys:
            array = np.asarray(
                [float(row[metric]) for row in values if row.get(metric) is not None],
                dtype=float,
            )
            if len(array):
                result[f"{metric}_mean"] = float(array.mean())
                result[f"{metric}_std"] = float(array.std(ddof=0))
        output.append(result)
    return output


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_root = Path(args.runs_root).expanduser().resolve()
    caches = {
        name: load_npz(path)
        for name, path in (
            ("blur_k9", args.blur_features),
            ("jpeg_q30", args.jpeg_features),
            ("resize_25", args.resize_features),
        )
    }
    adapter_rows = []
    residual_seed_rows = []
    cross_head_rows = []
    independent_rows = []
    error_group_rows = []
    transition_rows = []
    for seed in args.seeds:
        uniform, checkpoint = load_model(
            runs_root / f"uniform_seed{seed}" / "best.pt", device
        )
        restoration, _ = load_model(
            runs_root / f"restore_seed{seed}" / "best.pt", device
        )
        clean, _ = load_model(
            runs_root / f"clean_only_seed{seed}" / "best.pt", device
        )
        original_head = head_original(checkpoint, device)
        heads = {
            "original": original_head,
            "uniform": uniform.head,
            "restoration": restoration.head,
            "clean_only": clean.head,
        }
        for condition, data in caches.items():
            indices = {
                split: split_indices(checkpoint, split, data["video_ids"])
                for split in ("train", "validation", "test")
            }
            states = {
                split: build_states(
                    uniform,
                    restoration,
                    clean,
                    data,
                    index,
                    device,
                )
                for split, index in indices.items()
            }
            adapter_rows.extend(
                adapter_comparison_rows(
                    states["test"],
                    seed,
                    condition,
                    original_head,
                    uniform,
                    restoration,
                    clean,
                    args.threshold,
                )
            )
            for split in ("train", "validation", "test"):
                residual_seed_rows.extend(
                    {
                        "condition": condition,
                        **row,
                    }
                    for row in residual_rows(
                        states[split],
                        seed,
                        split,
                        uniform.head,
                        restoration.head,
                        original_head,
                    )
                )
                cross_head_rows.extend(
                    shared_head_rows(
                        states[split],
                        seed,
                        condition,
                        split,
                        heads,
                        args.threshold,
                    )
                )
                groups, transitions = error_conditioned_rows(
                    states[split],
                    seed,
                    split,
                    original_head,
                    uniform.head,
                    args.threshold,
                )
                for row in groups:
                    row["condition"] = condition
                for row in transitions:
                    row["condition"] = condition
                error_group_rows.extend(groups)
                transition_rows.extend(transitions)
            independent_rows.extend(
                independent_head_rows(
                    states["train"],
                    states["validation"],
                    states["test"],
                    seed,
                    condition,
                    args.independent_c,
                    args.threshold,
                )
            )
    write_csv(output_dir / "adapter_summary_seed.csv", adapter_rows)
    write_csv(
        output_dir / "adapter_summary.csv",
        aggregate(adapter_rows, ("condition", "split", "method")),
    )
    write_csv(output_dir / "residual_metrics_seed.csv", residual_seed_rows)
    write_csv(
        output_dir / "residual_metrics_summary.csv",
        aggregate(
            residual_seed_rows,
            ("condition", "split", "residual", "classifier", "group"),
        ),
    )
    write_csv(output_dir / "cross_head_seed.csv", cross_head_rows)
    write_csv(
        output_dir / "cross_head_summary.csv",
        aggregate(
            cross_head_rows,
            ("condition", "split", "head", "features"),
        ),
    )
    write_csv(output_dir / "independent_head.csv", independent_rows)
    write_csv(
        output_dir / "independent_head_summary.csv",
        aggregate(
            independent_rows,
            ("condition", "split", "features", "C"),
        ),
    )
    write_csv(output_dir / "error_groups_seed.csv", error_group_rows)
    write_csv(
        output_dir / "error_groups_summary.csv",
        aggregate(
            error_group_rows,
            ("split", "condition", "group"),
        ),
    )
    write_csv(output_dir / "error_transitions.csv", transition_rows)
    resource_audit = {
        "uniform_checkpoints": [
            str((runs_root / f"uniform_seed{seed}" / "best.pt").resolve())
            for seed in args.seeds
        ],
        "restoration_checkpoints": [
            str((runs_root / f"restore_seed{seed}" / "best.pt").resolve())
            for seed in args.seeds
        ],
        "clean_only_checkpoints": [
            str((runs_root / f"clean_only_seed{seed}" / "best.pt").resolve())
            for seed in args.seeds
        ],
        "historical_training_protocol": {
            "shared_split": True,
            "uniform_method": "uniform",
            "restoration_method": "restore",
            "restoration_additional_weights": {
                "restoration_weight": 5.0,
                "consistency_weight": 5.0,
            },
            "same_adapter_architecture": True,
        },
        "evaluation_scope": "Four fully degraded frames per video; no mask mixing.",
        "independent_head": {
            "estimator": "LogisticRegression",
            "C": args.independent_c,
            "class_weight": "balanced",
            "fit_split": "train",
            "selection": "No test-based tuning.",
        },
    }
    (output_dir / "resource_audit.json").write_text(
        json.dumps(resource_audit, indent=2),
        encoding="utf-8",
    )
    print(output_dir)


if __name__ == "__main__":
    main()
