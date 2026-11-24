import numpy as np
import torch

from src.residual_stage2 import (
    interpolate_features,
    reverse_intervention_start,
    selector_scores,
)
from src.residual_stage2_extended import (
    bootstrap_groups,
    evaluate_mask,
    mask_from_indices,
    random_mask,
    top_k_mask,
)
from src.adapter_mechanism_stage2 import transition_code, transition_group


def _bundle_and_head():
    torch.manual_seed(7)
    z0 = torch.nn.functional.normalize(torch.randn(6, 4, 32), dim=-1)
    zd = torch.nn.functional.normalize(torch.randn(6, 4, 32), dim=-1)
    za = torch.nn.functional.normalize(torch.randn(6, 4, 32), dim=-1)
    zc = torch.nn.functional.normalize(z0 + 0.05 * torch.randn_like(z0), dim=-1)
    head = torch.nn.Linear(32, 2)
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    bundle = {
        "z0": z0,
        "zd": zd,
        "za": za,
        "zc": zc,
        "R1": z0 - zd,
        "R2": z0 - za,
        "R3": zc - za,
        "labels": labels,
    }
    return bundle, head


def test_r3_reverse_uses_adapter_space_clean_endpoint():
    bundle, head = _bundle_and_head()
    assert torch.equal(reverse_intervention_start(bundle, "R2"), bundle["z0"])
    assert torch.equal(reverse_intervention_start(bundle, "R3"), bundle["zc"])
    full = torch.ones(bundle["za"].shape[-1], dtype=torch.bool)
    metrics, probability = evaluate_mask(
        reverse_intervention_start(bundle, "R3"),
        bundle["R3"],
        full,
        head,
        bundle["labels"],
        "reverse",
        0.5,
    )
    expected_logits = head(bundle["za"])
    expected_probability = expected_logits.softmax(-1)[..., 1].mean(-1)
    assert torch.allclose(probability, expected_probability, atol=1e-6)
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_zero_and_full_addback_are_numerically_exact():
    bundle, head = _bundle_and_head()
    zero = torch.zeros(bundle["za"].shape[-1], dtype=torch.bool)
    _, zero_probability = evaluate_mask(
        bundle["za"],
        bundle["R3"],
        zero,
        head,
        bundle["labels"],
        "add",
        0.5,
    )
    expected_adapted = head(bundle["za"]).softmax(-1)[..., 1].mean(-1)
    assert torch.allclose(zero_probability, expected_adapted, atol=1e-6)

    full = torch.ones(bundle["za"].shape[-1], dtype=torch.bool)
    _, full_probability = evaluate_mask(
        bundle["za"],
        bundle["R3"],
        full,
        head,
        bundle["labels"],
        "add",
        0.5,
    )
    expected_clean = head(bundle["zc"]).softmax(-1)[..., 1].mean(-1)
    assert torch.allclose(full_probability, expected_clean, atol=1e-6)


def test_interpolation_endpoints():
    bundle, _ = _bundle_and_head()
    alpha_zero = interpolate_features(bundle["za"], bundle["R3"], 0.0)
    alpha_one = interpolate_features(bundle["za"], bundle["R3"], 1.0)
    assert torch.allclose(alpha_zero, bundle["za"], atol=1e-6)
    assert torch.allclose(alpha_one, bundle["zc"], atol=1e-6)


def test_signed_selector_keeps_signed_top_k_and_is_deterministic():
    bundle, head = _bundle_and_head()
    scores = selector_scores(None, bundle, "R3", head)
    high = torch.tensor(2.0)
    low = torch.tensor(-3.0)
    scores["signed_discriminative"][0] = low
    scores["signed_discriminative"][1] = high
    first = top_k_mask(scores["signed_discriminative"], 2)
    second = top_k_mask(scores["signed_discriminative"], 2)
    assert torch.equal(first, second)
    assert first[0].item() is False
    assert first[1].item() is True


def test_random_masks_have_requested_count_and_repeatability():
    energy = torch.linspace(0.1, 1.0, 64)
    first, first_error = random_mask(
        64, 8, 2026, energy=energy, target_energy=4.0
    )
    second, second_error = random_mask(
        64, 8, 2026, energy=energy, target_energy=4.0
    )
    assert first.sum().item() == 8
    assert torch.equal(first, second)
    assert first_error == second_error


def test_group_bootstrap_keeps_group_together():
    groups = np.asarray(["a", "a", "b", "c"])
    base = np.asarray([0, 0, 0, 1], dtype=bool)
    new = np.asarray([1, 0, 1, 1], dtype=bool)
    result = bootstrap_groups(groups, base, new, iterations=200, seed=1)
    assert result["groups"] == 3
    assert result["videos"] == 4
    assert result["ci95_lower"] <= result["difference"] <= result["ci95_upper"]


def test_mask_from_indices_is_unique_boolean_mask():
    mask = mask_from_indices(torch.tensor([1, 3, 3]), 5)
    assert mask.dtype == torch.bool
    assert mask.tolist() == [False, True, False, True, False]


def test_error_transition_groups_cover_target_cases():
    assert (
        transition_group(True, False, True)
        == "A_successful_correction"
    )
    assert (
        transition_group(True, False, False)
        == "B_persistent_degradation_error"
    )
    assert (
        transition_group(True, True, False)
        == "C_adapter_introduced_error"
    )
    assert (
        transition_group(False, False, False)
        == "D_persistent_original_error"
    )
    assert transition_group(True, True, True) == "E_stable_correct"
    assert transition_code(True, False, True) == "101"
