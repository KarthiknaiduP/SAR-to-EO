"""Run SAR-to-EO inference from a trained checkpoint.

Example:
    python infer.py \
        --checkpoint sar2eo_outputs/model_5_final_fusion_gan/checkpoints/best.pt \
        --input sample_sar.png \
        --output generated_eo.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sar2eo_common import load_generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an EO-like RGB image from a SAR input image.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint best.pt or latest.pt.")
    parser.add_argument("--input", required=True, help="Path to input SAR image.")
    parser.add_argument("--output", required=True, help="Path where generated EO image will be saved.")
    parser.add_argument("--image_size", type=int, default=256, help="Inference size used by the trained models.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference even if CUDA/MPS is available.")
    return parser.parse_args()


def choose_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_sar_image(path: str | Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("L").resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32)[None, None] / 127.5 - 1.0
    return torch.from_numpy(array)


def save_rgb_tensor(tensor: torch.Tensor, path: str | Path) -> None:
    image = tensor.detach().cpu().clamp(-1, 1)
    array = ((image.squeeze(0) + 1.0) * 127.5).permute(1, 2, 0).numpy()
    array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(output)


def main() -> None:
    args = parse_args()
    device = choose_device(args.cpu)

    generator, payload = load_generator(args.checkpoint, device)
    sar = load_sar_image(args.input, args.image_size).to(device)

    with torch.inference_mode():
        prediction = generator(sar)

    save_rgb_tensor(prediction, args.output)

    config = payload.get("config", {})
    print(f"Saved generated EO image: {args.output}")
    print(f"Checkpoint epoch: {payload.get('epoch', 'unknown')}")
    print(f"Model: {config.get('run_name', config.get('model_name', 'unknown'))}")
    print(f"Device: {device}")


if __name__ == "__main__":
    main()
