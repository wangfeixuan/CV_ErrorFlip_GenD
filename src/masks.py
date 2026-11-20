from __future__ import annotations

import torch


def all_masks(frames: int = 4, device: torch.device | str | None = None) -> torch.Tensor:
    """Return every binary blur mask, least significant bit first."""
    values = torch.arange(2**frames, dtype=torch.long, device=device)
    shifts = torch.arange(frames, dtype=torch.long, device=device)
    return ((values[:, None] >> shifts[None, :]) & 1).to(torch.float32)


def masks_with_count(frames: int, blurred: int, device=None) -> torch.Tensor:
    masks = all_masks(frames, device=device)
    return masks[masks.sum(dim=1).to(torch.long) == blurred]


def compose_masks(
    clean: torch.Tensor, blurred: torch.Tensor, masks: torch.Tensor
) -> torch.Tensor:
    """Compose [B, F, D] feature pairs with [M, F] binary masks."""
    if clean.shape != blurred.shape or clean.ndim != 3:
        raise ValueError("clean/blurred must both have shape [B, F, D]")
    if masks.ndim != 2 or masks.shape[1] != clean.shape[1]:
        raise ValueError("masks must have shape [M, F]")
    mask = masks.to(clean.device, clean.dtype)
    return (
        clean[None, :, :, :] * (1.0 - mask[:, None, :, None])
        + blurred[None, :, :, :] * mask[:, None, :, None]
    )


def compose_frame_probabilities(
    clean: torch.Tensor, blurred: torch.Tensor, masks: torch.Tensor
) -> torch.Tensor:
    """Return one video probability per mask for cached per-frame probabilities."""
    if clean.shape != blurred.shape or clean.ndim != 2:
        raise ValueError("clean/blurred probabilities must have shape [B, F]")
    if masks.ndim != 2 or masks.shape[1] != clean.shape[1]:
        raise ValueError("masks must have shape [M, F]")
    mask = masks.to(clean.device, clean.dtype)
    frame_probabilities = (
        clean[None, :, :] * (1.0 - mask[:, None, :])
        + blurred[None, :, :] * mask[:, None, :]
    )
    return frame_probabilities.mean(dim=-1)
