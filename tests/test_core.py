import csv
import io
import tarfile

import numpy as np
import torch
from PIL import Image

from src.adapter import GenDAdapter
from src.data import (
    apply_group_manifest,
    assign_group_splits,
    index_zip,
    source_group,
)
from src.import_dfdc import (
    evenly_spaced_indices,
    read_metadata,
    source_group_id,
)
from src.import_wilddeepfake import import_shards
from src.masks import all_masks, masks_with_count
from src.metrics import binary_metrics


def test_mask_enumeration():
    masks = all_masks(4)
    assert masks.shape == (16, 4)
    assert len({tuple(row.tolist()) for row in masks}) == 16
    assert masks_with_count(4, 2).shape == (6, 4)
    assert torch.all(masks_with_count(4, 2).sum(dim=1) == 2)


def test_source_groups_and_splits_do_not_leak():
    records = []
    for index in range(100):
        label = "fake" if index % 2 else "real"
        video_id = (
            f"Celeb-synthesis__id{index // 10}_id{index % 10}_000{index % 4}"
            if label == "fake"
            else f"Celeb-real__id{index // 2}_000{index % 4}"
        )
        records.append(
            {
                "video_id": video_id,
                "label": label,
                "label_index": int(label == "fake"),
                "group_id": source_group(video_id, label),
            }
        )
    assigned = assign_group_splits(
        records,
        {"train": 0.7, "validation": 0.15, "test": 0.15},
        2026,
    )
    groups = {}
    for record in assigned:
        groups.setdefault(record["group_id"], set()).add(record["split"])
    assert all(len(splits) == 1 for splits in groups.values())
    assert set(record["split"] for record in assigned) == {
        "train",
        "validation",
        "test",
    }


def test_adapter_initializes_from_original_head():
    torch.manual_seed(0)
    features = torch.nn.functional.normalize(torch.randn(2, 4, 16), dim=-1)
    weight = torch.randn(2, 16)
    bias = torch.randn(2)
    model = GenDAdapter.from_pretrained_head(weight, bias, bottleneck=4)
    model.eval()
    with torch.no_grad():
        logits, probability = model(features)
    expected_logits = features @ weight.T + bias
    assert torch.allclose(logits, expected_logits, atol=1e-6)
    assert torch.allclose(probability, expected_logits.softmax(-1)[..., 1].mean(-1))


def test_metrics():
    labels = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.1, 0.4, 0.6, 0.9])
    metrics = binary_metrics(labels, probabilities)
    assert metrics["accuracy"] == 1.0
    assert metrics["auc"] == 1.0


def test_wilddeepfake_import_fixture(tmp_path):
    archive_path = tmp_path / "archives" / "real" / "1.tar.gz"
    archive_path.parent.mkdir(parents=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        for frame in range(5):
            image = Image.new("RGB", (224, 224), (frame, frame, frame))
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            payload = buffer.getvalue()
            info = tarfile.TarInfo(name=f"video_1/{frame:04d}.png")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    manifest = tmp_path / "shards.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["label", "path", "size_bytes"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "label": "real",
                "path": "deepfake_in_the_wild/real_train/1.tar.gz",
                "size_bytes": archive_path.stat().st_size,
            }
        )
    output_zip = tmp_path / "faces.zip"
    output_manifest = tmp_path / "sequences.csv"
    import_shards(
        type(
            "Args",
            (),
            {
                "shard_manifest": str(manifest),
                "archives_dir": str(tmp_path / "archives"),
                "output_zip": str(output_zip),
                "output_manifest": str(output_manifest),
                "frames_per_sequence": 4,
            },
        )()
    )
    indexed = index_zip(output_zip)
    assert len(indexed) == 1
    assert indexed[0]["label"] == "real"
    assert indexed[0]["frame_count"] == 4


def test_dfdc_metadata_and_source_grouping(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        """
        {
          "original.mp4": {"label": "REAL", "original": null},
          "fake-a.mp4": {"label": "FAKE", "original": "original.mp4"},
          "fake-b.mp4": {"label": "FAKE", "original": "original.mp4"}
        }
        """,
        encoding="utf-8",
    )
    metadata = read_metadata(metadata_path)
    assert metadata["original.mp4"]["label"] == "real"
    assert metadata["fake-a.mp4"]["label"] == "fake"
    assert source_group_id(None, "original.mp4") == source_group_id(
        "original.mp4", "fake-a.mp4"
    )
    assert source_group_id("original.mp4", "fake-a.mp4") == source_group_id(
        "original.mp4", "fake-b.mp4"
    )
    assert evenly_spaced_indices(10, 4) == [0, 3, 6, 9]


def test_group_manifest_overrides_zip_groups(tmp_path):
    records = [
        {"video_id": "fake-a", "label": "fake", "group_id": "fake-video-a"},
        {"video_id": "real-source", "label": "real", "group_id": "real-video-source"},
    ]
    manifest = tmp_path / "groups.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id", "label", "group_id"])
        writer.writeheader()
        writer.writerow(
            {"video_id": "fake-a", "label": "fake", "group_id": "dfdc-source-1"}
        )
        writer.writerow(
            {"video_id": "real-source", "label": "real", "group_id": "dfdc-source-1"}
        )
    updated = apply_group_manifest(records, manifest)
    assert {record["group_id"] for record in updated} == {"dfdc-source-1"}


def test_multi_label_source_groups_do_not_leak():
    records = []
    for group in range(20):
        group_id = f"dfdc-source-{group}"
        records.append(
            {
                "video_id": f"real-{group}",
                "label": "real",
                "group_id": group_id,
            }
        )
        for fake in range(3):
            records.append(
                {
                    "video_id": f"fake-{group}-{fake}",
                    "label": "fake",
                    "group_id": group_id,
                }
            )
    assigned = assign_group_splits(
        records,
        {"train": 0.85, "validation": 0.15, "test": 0.0},
        2026,
    )
    group_splits = {}
    for record in assigned:
        group_splits.setdefault(record["group_id"], set()).add(record["split"])
    assert all(len(splits) == 1 for splits in group_splits.values())
    assert {record["split"] for record in assigned} == {"train", "validation"}
