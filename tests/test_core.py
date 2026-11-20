import numpy as np
import torch

from src.adapter import GenDAdapter
from src.data import assign_group_splits, source_group
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
