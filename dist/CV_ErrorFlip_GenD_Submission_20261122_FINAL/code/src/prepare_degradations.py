from __future__ import annotations

import argparse
import io
import json
import os
import uuid
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-zip", required=True)
    parser.add_argument("--output-zip", required=True)
    parser.add_argument(
        "--kind", choices=("blur_k9", "jpeg_q30", "resize_25"), required=True
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def degrade(image: Image.Image, kind: str) -> Image.Image:
    image = image.convert("RGB")
    if kind == "blur_k9":
        import cv2
        import numpy as np

        array = np.asarray(image)
        blurred = cv2.GaussianBlur(
            array, (9, 9), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_REFLECT_101
        )
        return Image.fromarray(blurred, mode="RGB")
    if kind == "jpeg_q30":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=30, subsampling=2, optimize=False)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB").copy()
    if kind == "resize_25":
        size = (max(1, image.width // 4), max(1, image.height // 4))
        return image.resize(size, Image.Resampling.BICUBIC)
    raise ValueError(kind)


def main() -> None:
    args = parse_args()
    source = Path(args.original_zip).expanduser().resolve()
    output = Path(args.output_zip).expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".{uuid.uuid4().hex[:8]}.tmp")
    count = 0
    try:
        with ZipFile(source) as input_zip, ZipFile(
            temporary, "w", compression=ZIP_STORED
        ) as output_zip:
            names = sorted(
                name for name in input_zip.namelist() if name.lower().endswith(".png")
            )
            for name in tqdm(names, desc=args.kind):
                with Image.open(io.BytesIO(input_zip.read(name))) as image:
                    transformed = degrade(image, args.kind)
                buffer = io.BytesIO()
                transformed.save(buffer, format="PNG", optimize=False)
                output_zip.writestr(name, buffer.getvalue())
                count += 1
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    metadata = {
        "source_zip": str(source),
        "output_zip": str(output),
        "kind": args.kind,
        "images": count,
        "protocol": {
            "blur_k9": "OpenCV GaussianBlur 9x9, sigma=0, BORDER_REFLECT_101",
            "jpeg_q30": "Pillow JPEG quality=30, subsampling=2, then decoded and stored as PNG",
            "resize_25": "Pillow bicubic resize to floor(width/4) x floor(height/4), stored as PNG",
        }[args.kind],
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
