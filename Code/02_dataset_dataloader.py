"""
Notebook 2: Clean PyTorch Dataset + DataLoader validation.

Run in Kaggle:
    python 02_dataset_dataloader.py

This script uses the manifest created by 01_eda_dataset_exploration.py.
It verifies tensor shapes, normalization, train/val/test counts, and writes
a visual batch preview.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


DEFAULT_MANIFEST = "/kaggle/working/sar2eo_outputs/manifest.csv"
DEFAULT_OUT = "/kaggle/working/sar2eo_outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out_dir", default=DEFAULT_OUT)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=256)
    return parser.parse_args()


def load_manifest(path: Path, split: str | None = None) -> list[dict[str, str]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if split:
        rows = [r for r in rows if r["split"] == split]
    return rows


def load_sar(path: str, image_size: int) -> torch.Tensor:
    img = Image.open(path).convert("L").resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr[None, :, :]
    return torch.from_numpy(arr * 2.0 - 1.0)


def load_rgb(path: str, image_size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr * 2.0 - 1.0)


def augment_pair(sar: torch.Tensor, eo: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if random.random() < 0.5:
        sar = torch.flip(sar, dims=[2])
        eo = torch.flip(eo, dims=[2])
    if random.random() < 0.5:
        sar = torch.flip(sar, dims=[1])
        eo = torch.flip(eo, dims=[1])
    k = random.randint(0, 3)
    if k:
        sar = torch.rot90(sar, k, dims=[1, 2])
        eo = torch.rot90(eo, k, dims=[1, 2])
    return sar.contiguous(), eo.contiguous()


class SarOpticalDataset(Dataset):
    def __init__(self, manifest: str | Path, split: str, image_size: int = 256, augment: bool = False, limit: int | None = None):
        self.rows = load_manifest(Path(manifest), split)
        if limit:
            self.rows = self.rows[:limit]
        self.image_size = image_size
        self.augment = augment
        if not self.rows:
            raise ValueError(f"No rows found for split={split}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        sar = load_sar(row["sar_path"], self.image_size)
        eo = load_rgb(row["eo_path"], self.image_size)
        if self.augment:
            sar, eo = augment_pair(sar, eo)
        return {
            "sar": sar,
            "eo": eo,
            "sar_path": row["sar_path"],
            "eo_path": row["eo_path"],
            "scene_id": row["scene_id"],
            "terrain": row.get("terrain", ""),
        }


def denorm(t: torch.Tensor) -> np.ndarray:
    t = (t.detach().cpu().clamp(-1, 1) + 1.0) / 2.0
    if t.shape[0] == 1:
        return t[0].numpy()
    return t.permute(1, 2, 0).numpy()


def save_batch_preview(batch: dict, out_path: Path, max_items: int = 6) -> None:
    n = min(max_items, batch["sar"].shape[0])
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    if n == 1:
        axes = np.asarray([axes])
    for i in range(n):
        axes[i, 0].imshow(denorm(batch["sar"][i]), cmap="gray")
        axes[i, 0].set_title("SAR input [-1, 1]")
        axes[i, 1].imshow(denorm(batch["eo"][i]))
        axes[i, 1].set_title("EO target [-1, 1]")
        for ax in axes[i]:
            ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = load_manifest(manifest)
    print("Split counts:")
    for split in ["train", "val", "test"]:
        print(split, sum(r["split"] == split for r in all_rows))

    ds = SarOpticalDataset(manifest, args.split, args.image_size, augment=(args.split == "train"))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    batch = next(iter(loader))

    print("\nBatch checks")
    print("SAR shape:", tuple(batch["sar"].shape), "range:", float(batch["sar"].min()), float(batch["sar"].max()))
    print("EO shape:", tuple(batch["eo"].shape), "range:", float(batch["eo"].min()), float(batch["eo"].max()))
    print("Example scene:", batch["scene_id"][0])
    print("Example SAR:", batch["sar_path"][0])

    preview_path = out_dir / "dataloader_batch_preview.png"
    save_batch_preview(batch, preview_path)
    print("Batch preview:", preview_path)


if __name__ == "__main__":
    main()
