from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import uuid
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .constants import PROJECT_ROOT
from .data import apply_group_manifest, paired_zip_indexes, write_manifest
from .gend_model import GenDEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-zip", required=True)
    parser.add_argument("--blur-zip", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--group-manifest",
        help="CSV with video_id,label,group_id for cross-dataset source grouping",
    )
    parser.add_argument("--skip-model-hash", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(args: argparse.Namespace) -> Path:
    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
    records = paired_zip_indexes(args.original_zip, args.blur_zip)
    if args.group_manifest:
        records = apply_group_manifest(records, args.group_manifest)
    bad = [record["video_id"] for record in records if record["frame_count"] != 4]
    if bad:
        raise ValueError(f"Expected four frames per video; bad records: {bad[:10]}")

    encoder = GenDEncoder(
        args.model, device=args.device, verify_hash=not args.skip_model_hash
    )
    original_features: list[np.ndarray] = []
    blur_features: list[np.ndarray] = []
    base_original_logits: list[np.ndarray] = []
    base_blur_logits: list[np.ndarray] = []
    with ZipFile(args.original_zip) as original_zip, ZipFile(
        args.blur_zip
    ) as blur_zip:
        for start in tqdm(range(0, len(records), args.batch_size), desc="Extract"):
            batch_records = records[start : start + args.batch_size]
            original_images, blur_images = [], []
            for record in batch_records:
                for original_name, blur_name in zip(
                    record["frame_names"], record["blur_frame_names"]
                ):
                    with Image.open(io.BytesIO(original_zip.read(original_name))) as image:
                        original_images.append(image.convert("RGB").copy())
                    with Image.open(io.BytesIO(blur_zip.read(blur_name))) as image:
                        blur_images.append(image.convert("RGB").copy())
            clean_z, clean_logits = encoder.encode_pil_images(original_images)
            blur_z, blur_logits = encoder.encode_pil_images(blur_images)
            original_features.append(clean_z.reshape(len(batch_records), 4, 1024).numpy())
            blur_features.append(blur_z.reshape(len(batch_records), 4, 1024).numpy())
            base_original_logits.append(
                clean_logits.reshape(len(batch_records), 4, 2).numpy()
            )
            base_blur_logits.append(
                blur_logits.reshape(len(batch_records), 4, 2).numpy()
            )
            print(
                f"Extracted {min(start + len(batch_records), len(records))}/{len(records)} videos",
                flush=True,
            )

    head_weight = encoder.head.weight.detach().cpu().numpy().astype(np.float32)
    head_bias = encoder.head.bias.detach().cpu().numpy().astype(np.float32)
    arrays = {
        "features_original": np.concatenate(original_features).astype(np.float32),
        "features_blur": np.concatenate(blur_features).astype(np.float32),
        "base_logits": np.concatenate(base_original_logits).astype(np.float32),
        "base_logits_blur": np.concatenate(base_blur_logits).astype(np.float32),
        "labels": np.asarray([record["label"] for record in records]),
        "label_index": np.asarray(
            [record["label_index"] for record in records], dtype=np.int8
        ),
        "video_ids": np.asarray([record["video_id"] for record in records]),
        "group_ids": np.asarray([record["group_id"] for record in records]),
        "head_weight": head_weight,
        "head_bias": head_bias,
    }
    metadata = {
        "model_path": str(Path(args.model).resolve()),
        "model_sha256": (
            None if args.skip_model_hash else file_sha256(args.model)
        ),
        "original_zip": str(Path(args.original_zip).resolve()),
        "original_zip_sha256": file_sha256(args.original_zip),
        "blur_zip": str(Path(args.blur_zip).resolve()),
        "blur_zip_sha256": file_sha256(args.blur_zip),
        "group_manifest": (
            str(Path(args.group_manifest).resolve())
            if args.group_manifest
            else None
        ),
        "videos": len(records),
        "frames_per_video": 4,
        "feature_dim": 1024,
        "inventory": {
            "videos": len(records),
            "real": sum(record["label"] == "real" for record in records),
            "fake": sum(record["label"] == "fake" for record in records),
        },
        "code_root": str(PROJECT_ROOT),
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".{uuid.uuid4().hex[:8]}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, output)
    write_manifest(records, output.with_suffix(".manifest.csv"))
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved feature cache: {output}", flush=True)
    return output


def main() -> None:
    extract(parse_args())


if __name__ == "__main__":
    main()
