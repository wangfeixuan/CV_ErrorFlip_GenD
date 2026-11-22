from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from .constants import EXPECTED_MODEL_SHA256, EXPECTED_MODEL_SIZE


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model(path: str | Path, verify_hash: bool = True) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size != EXPECTED_MODEL_SIZE:
        raise ValueError(f"GenD weight size is {size:,}, expected {EXPECTED_MODEL_SIZE:,}")
    if verify_hash:
        actual = sha256(path)
        if actual != EXPECTED_MODEL_SHA256:
            raise ValueError(
                f"GenD SHA256 mismatch: expected {EXPECTED_MODEL_SHA256}, got {actual}"
            )
    return path


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


class GenDEncoder:
    """The exact published vision encoder/classifier structure used by GenD."""

    def __init__(
        self,
        model_path: str | Path,
        device: str | torch.device = "auto",
        verify_hash: bool = True,
    ) -> None:
        from safetensors import safe_open
        from transformers import CLIPVisionConfig, CLIPVisionModel

        self.device = (
            choose_device(device) if isinstance(device, str) else torch.device(device)
        )
        self.path = verify_model(model_path, verify_hash=verify_hash)
        config = CLIPVisionConfig(
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=24,
            num_attention_heads=16,
            image_size=224,
            patch_size=14,
            hidden_act="quick_gelu",
            layer_norm_eps=1e-5,
            attention_dropout=0.0,
        )
        vision_wrapper = CLIPVisionModel(config)
        # Transformers 4.x returns a wrapper with `.vision_model`; 5.x returns
        # the CLIP vision transformer directly.
        self.vision = getattr(vision_wrapper, "vision_model", vision_wrapper)
        self.head = torch.nn.Linear(1024, 2)
        expected: set[str] = set()
        with safe_open(str(self.path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            with torch.no_grad():
                for name, parameter in self.vision.named_parameters():
                    key = "feature_extractor.vision_model." + name
                    expected.add(key)
                    _copy_tensor(handle, key, parameter, keys)
                for name, parameter in self.head.named_parameters():
                    key = "model.linear." + name
                    expected.add(key)
                    _copy_tensor(handle, key, parameter, keys)
            extra = {
                key
                for key in keys
                if key.startswith("feature_extractor.vision_model.")
                or key.startswith("model.linear.")
            } - expected
            if extra:
                raise RuntimeError(f"Unexpected GenD checkpoint keys: {sorted(extra)[:5]}")
        self.vision.to(self.device).eval()
        self.head.to(self.device).eval()
        self.processor = _make_processor()

    @torch.inference_mode()
    def encode_pil_images(self, images) -> tuple[torch.Tensor, torch.Tensor]:
        pixel_values = self.processor(images=images, return_tensors="pt")[
            "pixel_values"
        ].to(self.device)
        pooled = self.vision(pixel_values).pooler_output.float()
        features = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        logits = self.head(features)
        return features.cpu(), logits.cpu()


def _copy_tensor(handle, key: str, parameter: torch.Tensor, keys: set[str]) -> None:
    if key not in keys:
        raise RuntimeError(f"Missing GenD checkpoint key: {key}")
    value = handle.get_tensor(key)
    if tuple(value.shape) != tuple(parameter.shape):
        raise RuntimeError(
            f"Shape mismatch for {key}: {tuple(value.shape)} vs {tuple(parameter.shape)}"
        )
    parameter.copy_(value)


def _make_processor():
    from PIL import Image
    from transformers import CLIPImageProcessor

    return CLIPImageProcessor(
        do_resize=True,
        size={"shortest_edge": 224},
        resample=Image.Resampling.BICUBIC,
        do_center_crop=True,
        crop_size={"height": 224, "width": 224},
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
    )
