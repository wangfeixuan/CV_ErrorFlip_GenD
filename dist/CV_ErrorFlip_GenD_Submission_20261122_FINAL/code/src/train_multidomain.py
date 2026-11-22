from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from .adapter import GenDAdapter
from .constants import DEFAULT_CONFIG
from .gend_model import choose_device
from .masks import compose_frame_probabilities, masks_with_count
from .train import (
    binary_cross_entropy,
    evaluate_loss_and_metrics,
    load_features,
    make_records,
    mask_losses_and_probabilities,
    select_flip_weighted_loss,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--celeb-features", required=True)
    parser.add_argument("--dfdc-features", required=True)
    parser.add_argument(
        "--method",
        choices=("clean_only", "uniform", "flip_weighted"),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--config", default="configs/multidomain.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--batch-size-per-domain", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--bottleneck", type=int)
    parser.add_argument("--identity-reg-weight", type=float)
    parser.add_argument("--variant-name")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def split_domain_data(
    data: dict,
    ratios: dict,
    split_seed: int,
) -> tuple[list[dict], np.ndarray]:
    records = make_records(data, split_seed, ratios)
    split_by_video = {row["video_id"]: row["split"] for row in records}
    splits = np.asarray(
        [split_by_video[str(video_id)] for video_id in data["video_ids"]]
    )
    return records, splits


def balanced_sample(
    indices: torch.Tensor,
    labels: torch.Tensor,
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    real = indices[labels[indices] == 0]
    fake = indices[labels[indices] == 1]
    if not len(real) or not len(fake):
        raise ValueError("Each domain needs both real and fake training videos")
    real_count = count // 2
    fake_count = count - real_count
    real_pick = real[
        torch.randint(
            len(real),
            (real_count,),
            generator=generator,
            device=indices.device,
        )
    ]
    fake_pick = fake[
        torch.randint(
            len(fake),
            (fake_count,),
            generator=generator,
            device=indices.device,
        )
    ]
    return torch.cat((real_pick, fake_pick))


def prepare_domain(data: dict, splits: np.ndarray, device: torch.device) -> dict:
    result = {
        "clean": torch.from_numpy(data["features_original"]).to(device),
        "blurred": torch.from_numpy(data["features_blur"]).to(device),
        "labels": torch.from_numpy(
            data["label_index"].astype(np.int64)
        ).to(device),
    }
    split_indices = {
        name: torch.from_numpy(
            np.flatnonzero(splits == name).astype(np.int64)
        ).to(device)
        for name in ("train", "validation", "test")
    }
    split_indices["splits"] = splits
    result.update(split_indices)
    return result


def concat_domain_batch(domain_batches: list[dict], key: str) -> torch.Tensor:
    return torch.cat([batch[key] for batch in domain_batches], dim=0)


def evaluate_domain(
    model: GenDAdapter,
    domain: dict,
    masks: torch.Tensor,
    threshold: float,
) -> dict:
    index = domain["validation"]
    labels = domain["labels"][index].float()
    real_count = int((labels == 0).sum())
    fake_count = int((labels == 1).sum())
    validation_fake_weight = real_count / max(fake_count, 1)
    return evaluate_loss_and_metrics(
        model,
        domain["clean"][index],
        domain["blurred"][index],
        labels,
        masks,
        threshold,
        validation_fake_weight,
    )


def train(args: argparse.Namespace) -> Path:
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    training = dict(config["training"])
    model_config = dict(config["model"])
    overrides = {
        "epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
    }
    training.update(
        {key: value for key, value in overrides.items() if value is not None}
    )
    if args.batch_size_per_domain is not None:
        training["batch_size_per_domain"] = args.batch_size_per_domain
    if args.bottleneck is not None:
        model_config["bottleneck"] = args.bottleneck
    if args.identity_reg_weight is not None:
        training["identity_reg_weight"] = args.identity_reg_weight
    variant_name = args.variant_name or args.method

    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output} is not empty; pass --overwrite to reuse it")
    output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = choose_device(args.device)
    ratios = config["data"]["split_ratios"]
    split_seed = int(config["seed"])

    domain_data = {
        "celebdf": load_features(args.celeb_features),
        "dfdc": load_features(args.dfdc_features),
    }
    domain_records = {}
    domain_tensors = {}
    for domain_name, data in domain_data.items():
        records, splits = split_domain_data(data, ratios, split_seed)
        domain_records[domain_name] = records
        domain_tensors[domain_name] = prepare_domain(data, splits, device)

    all_record_ids = [
        record["video_id"]
        for records in domain_records.values()
        for record in records
    ]
    if len(all_record_ids) != len(set(all_record_ids)):
        raise ValueError("Video IDs collide across domains")
    split_by_video = {
        record["video_id"]: record["split"]
        for records in domain_records.values()
        for record in records
    }
    group_by_video = {
        record["video_id"]: record["group_id"]
        for records in domain_records.values()
        for record in records
    }

    batch_per_domain = int(training["batch_size_per_domain"])
    steps_per_epoch = max(
        math.ceil(
            len(domain_tensors[name]["train"])
            / max(1, batch_per_domain)
        )
        for name in domain_tensors
    )
    train_masks = masks_with_count(
        4, int(training.get("blurred_frames", 2)), device=device
    )
    evaluation_masks = masks_with_count(4, 2, device=device)

    for name, domain in domain_tensors.items():
        base_clean = torch.softmax(
            torch.from_numpy(
                domain_data[name]["base_logits"]
            ).to(device),
            dim=-1,
        )[..., 1].mean(dim=1)
        base_blur_frames = torch.softmax(
            torch.from_numpy(
                domain_data[name]["base_logits_blur"]
            ).to(device),
            dim=-1,
        )[..., 1]
        domain["base_clean_probability"] = base_clean
        domain["base_mask_probability"] = compose_frame_probabilities(
            torch.softmax(
                torch.from_numpy(
                    domain_data[name]["base_logits"]
                ).to(device),
                dim=-1,
            )[..., 1],
            base_blur_frames,
            train_masks,
        ).T

    head_data = domain_data["celebdf"]
    model = GenDAdapter.from_pretrained_head(
        torch.from_numpy(head_data["head_weight"]),
        torch.from_numpy(head_data["head_bias"]),
        bottleneck=int(model_config["bottleneck"]),
        dropout=float(model_config["dropout"]),
        scale_init=float(model_config["scale_init"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    best_loss = math.inf
    best_epoch = -1
    stale_epochs = 0
    history = []

    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        epoch_loss = 0.0
        steps = 0
        for _ in range(steps_per_epoch):
            batches = []
            for domain_name, domain in domain_tensors.items():
                sampled = balanced_sample(
                    domain["train"],
                    domain["labels"],
                    batch_per_domain,
                    generator,
                )
                batches.append(
                    {
                        "index": sampled,
                        "clean": domain["clean"][sampled],
                        "blurred": domain["blurred"][sampled],
                        "labels": domain["labels"][sampled].float(),
                        "base_clean_probability": domain[
                            "base_clean_probability"
                        ][sampled],
                        "base_mask_probability": domain[
                            "base_mask_probability"
                        ][sampled],
                    }
                )
            clean = concat_domain_batch(batches, "clean")
            blurred = concat_domain_batch(batches, "blurred")
            labels = concat_domain_batch(batches, "labels")

            if args.method == "clean_only":
                degraded_loss = None
            else:
                losses, _ = mask_losses_and_probabilities(
                    model, clean, blurred, labels, train_masks
                )
                if args.method == "uniform":
                    degraded_loss = losses.mean()
                else:
                    base_clean = concat_domain_batch(
                        batches, "base_clean_probability"
                    )
                    base_mask = torch.cat(
                        [batch["base_mask_probability"] for batch in batches],
                        dim=0,
                    )
                    degraded_loss = select_flip_weighted_loss(
                        losses,
                        base_clean,
                        base_mask,
                        labels,
                        float(training["threshold"]),
                        float(training["flip_weight"]),
                    )

            _, clean_probability, adapted = model(
                clean, return_features=True
            )
            clean_loss = binary_cross_entropy(clean_probability, labels)
            if degraded_loss is None:
                loss = (
                    float(training["clean_loss_weight"])
                    + float(training["degraded_loss_weight"])
                ) * clean_loss
            else:
                loss = (
                    float(training["clean_loss_weight"]) * clean_loss
                    + float(training["degraded_loss_weight"]) * degraded_loss
                )
            if float(training.get("identity_reg_weight", 0.0)):
                identity_penalty = (
                    1.0
                    - torch.nn.functional.cosine_similarity(
                        adapted, clean, dim=-1
                    )
                ).mean()
                loss = loss + float(
                    training["identity_reg_weight"]
                ) * identity_penalty

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip"])
            )
            optimizer.step()
            epoch_loss += float(loss.item())
            steps += 1

        validation = {
            name: evaluate_domain(
                model,
                domain,
                evaluation_masks,
                float(training["threshold"]),
            )
            for name, domain in domain_tensors.items()
        }
        validation_loss = float(
            np.mean([result["loss"] for result in validation.values()])
        )
        row = {
            "epoch": epoch,
            "train_loss": epoch_loss / steps,
            "validation_loss": validation_loss,
            "celebdf_clean_accuracy": validation["celebdf"]["clean"]["accuracy"],
            "celebdf_mask_accuracy": validation["celebdf"][
                "degraded_mean_accuracy"
            ],
            "dfdc_clean_accuracy": validation["dfdc"]["clean"]["accuracy"],
            "dfdc_mask_accuracy": validation["dfdc"]["degraded_mean_accuracy"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "method": variant_name,
                    "base_method": args.method,
                    "seed": args.seed,
                    "split_seed": split_seed,
                    "blurred_frames": int(training.get("blurred_frames", 2)),
                    "epoch": epoch,
                    "validation": validation,
                    "split_by_video": split_by_video,
                    "group_by_video": group_by_video,
                    "feature_files": {
                        "celebdf": str(Path(args.celeb_features).resolve()),
                        "dfdc": str(Path(args.dfdc_features).resolve()),
                    },
                    "head_weight": head_data["head_weight"],
                    "head_bias": head_data["head_bias"],
                    "config": {
                        **config,
                        "training": training,
                        "model": model_config,
                        "method": variant_name,
                        "base_method": args.method,
                        "seed": args.seed,
                    },
                },
                output / "best.pt",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= int(training["patience"]):
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    split_rows = []
    for domain_name, records in domain_records.items():
        for record in records:
            split_rows.append({**record, "domain": domain_name})
    with (output / "split_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(split_rows[0]))
        writer.writeheader()
        writer.writerows(split_rows)
    (output / "run.json").write_text(
        json.dumps(
            {
                "method": variant_name,
                "base_method": args.method,
                "seed": args.seed,
                "split_seed": split_seed,
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "domain_split_summary": {
                    name: {
                        split: sum(
                            record["split"] == split for record in records
                        )
                        for split in ("train", "validation", "test")
                    }
                    for name, records in domain_records.items()
                },
                "feature_files": {
                    "celebdf": str(Path(args.celeb_features).resolve()),
                    "dfdc": str(Path(args.dfdc_features).resolve()),
                },
                "device": str(device),
                "config": {
                    **config,
                    "training": training,
                    "model": model_config,
                    "method": variant_name,
                    "base_method": args.method,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output / "best.pt"


def main() -> None:
    print(train(parse_args()))


if __name__ == "__main__":
    main()
