"""Evaluate every trained model on the same paired test samples.

Metrics: PSNR, SSIM, MAE, LPIPS, FID, parameter count, and inference time.
LPIPS/FID are reported as unavailable instead of silently replaced when their
dependencies or pretrained weights cannot be loaded.

Kaggle:
    python 08_evaluate_all_models.py --limit 1000
Specific checkpoints:
    python 08_evaluate_all_models.py --checkpoints /path/a.pt /path/b.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sar2eo_common import (
    DEFAULT_MANIFEST,
    DEFAULT_OUT,
    SarOpticalDataset,
    load_generator,
    tensor_to_image,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="*", default=None)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out_dir", default=DEFAULT_OUT)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--eval_name", default=None)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--skip_lpips", action="store_true")
    parser.add_argument("--skip_fid", action="store_true")
    parser.add_argument("--no_save_predictions", action="store_true")
    return parser.parse_args()


def discover_checkpoints(out_dir: Path) -> list[Path]:
    expected = [
        "model_1_unet_l1",
        "model_2_pix2pix",
        "model_3_pix2pix_perceptual",
        "model_4_attention_resunet",
        "model_5_final_fusion_gan",
    ]
    paths = [out_dir / name / "checkpoints" / "best.pt" for name in expected]
    return [path for path in paths if path.exists()]


def make_lpips(device: torch.device):
    try:
        import lpips

        return lpips.LPIPS(net="alex").to(device).eval()
    except Exception as error:
        print(f"LPIPS unavailable: {error!r}")
        return None


def make_fid(device: torch.device):
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance

        return FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    except Exception as error:
        print(f"FID unavailable: {error!r}")
        return None


def save_rgb(tensor: torch.Tensor, path: Path) -> None:
    array = ((tensor.detach().cpu().clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0)
    Image.fromarray(array.numpy().round().astype(np.uint8), "RGB").save(path)


def per_image_metrics(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float, float]:
    pred = tensor_to_image(prediction).astype(np.float32)
    real = tensor_to_image(target).astype(np.float32)
    mse = float(np.mean((pred - real) ** 2))
    psnr = float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)
    ssim = float(structural_similarity(real, pred, channel_axis=2, data_range=1.0))
    mae = float(np.mean(np.abs(pred - real)))
    return psnr, ssim, mae


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_checkpoint(
    checkpoint: Path,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    eval_root: Path,
) -> tuple[dict, list[dict]]:
    generator, payload = load_generator(checkpoint, device)
    config = payload["config"]
    run_name = config["run_name"]
    model_dir = eval_root / run_name
    prediction_dir = model_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    lpips_model = None if args.skip_lpips else make_lpips(device)
    fid = None if args.skip_fid else make_fid(device)
    rows: list[dict] = []
    elapsed, image_count = 0.0, 0

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"evaluate {run_name}"):
            sar = batch["sar"].to(device, non_blocking=True)
            target = batch["eo"].to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            prediction = generator(sar)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - started
            image_count += len(sar)
            lpips_values = (
                lpips_model(prediction, target).flatten().detach().cpu().tolist()
                if lpips_model is not None
                else [float("nan")] * len(sar)
            )
            if fid is not None:
                fid.update((target + 1) / 2, real=True)
                fid.update((prediction + 1) / 2, real=False)
            for index in range(len(sar)):
                psnr, ssim, mae = per_image_metrics(prediction[index], target[index])
                source_name = Path(batch["sar_path"][index]).stem
                filename = f"{len(rows):06d}_{source_name}.png"
                if not args.no_save_predictions:
                    save_rgb(prediction[index], prediction_dir / filename)
                rows.append(
                    {
                        "model": run_name,
                        "index": len(rows),
                        "scene_id": batch["scene_id"][index],
                        "terrain": batch["terrain"][index],
                        "sar_path": batch["sar_path"][index],
                        "eo_path": batch["eo_path"][index],
                        "prediction_path": str(prediction_dir / filename),
                        "psnr": psnr,
                        "ssim": ssim,
                        "mae": mae,
                        "lpips": lpips_values[index],
                    }
                )

    fid_value = float("nan")
    if fid is not None:
        try:
            fid_value = float(fid.compute().detach().cpu())
        except Exception as error:
            print(f"Could not compute FID for {run_name}: {error!r}")
    finite_lpips = [row["lpips"] for row in rows if math.isfinite(row["lpips"])]
    summary = {
        "model": run_name,
        "model_name": config["model_name"],
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": payload["epoch"],
        "count": len(rows),
        "parameters_millions": sum(parameter.numel() for parameter in generator.parameters()) / 1e6,
        "psnr": float(np.mean([row["psnr"] for row in rows])),
        "ssim": float(np.mean([row["ssim"] for row in rows])),
        "mae": float(np.mean([row["mae"] for row in rows])),
        "lpips": float(np.mean(finite_lpips)) if finite_lpips else float("nan"),
        "fid": fid_value,
        "milliseconds_per_image": elapsed * 1000 / max(image_count, 1),
    }
    write_csv(rows, model_dir / "per_image_metrics.csv")
    (model_dir / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=True))
    return summary, rows


def terrain_summary(all_rows: list[dict]) -> list[dict]:
    groups: defaultdict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in all_rows:
        groups[(row["model"], row["terrain"])].append(row)
    output = []
    for (model, terrain), rows in sorted(groups.items()):
        lpips_values = [row["lpips"] for row in rows if math.isfinite(row["lpips"])]
        output.append(
            {
                "model": model,
                "terrain": terrain,
                "count": len(rows),
                "psnr": np.mean([row["psnr"] for row in rows]),
                "ssim": np.mean([row["ssim"] for row in rows]),
                "mae": np.mean([row["mae"] for row in rows]),
                "lpips": np.mean(lpips_values) if lpips_values else float("nan"),
            }
        )
    return output


def plot_comparison(summary: list[dict], path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    metrics = [("psnr", "PSNR (higher is better)"), ("ssim", "SSIM (higher is better)"), ("lpips", "LPIPS (lower is better)"), ("fid", "FID (lower is better)")]
    labels = [row["model"].replace("model_", "M").replace("_", " ") for row in summary]
    colors = ["#287271", "#E9C46A", "#E76F51", "#2A9D8F", "#6D597A"]
    for axis, (metric, title) in zip(axes.flat, metrics):
        values = [row[metric] for row in summary]
        axis.bar(labels, values, color=colors[: len(values)])
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def qualitative_grid(all_rows: list[dict], path: Path, sample_count: int = 5) -> None:
    models = list(dict.fromkeys(row["model"] for row in all_rows))
    by_model = {model: {row["index"]: row for row in all_rows if row["model"] == model} for model in models}
    common = sorted(set.intersection(*(set(rows) for rows in by_model.values())))[:sample_count]
    if not common:
        return
    figure, axes = plt.subplots(len(common), len(models) + 2, figsize=(3 * (len(models) + 2), 3 * len(common)), squeeze=False)
    for row_index, index in enumerate(common):
        reference = by_model[models[0]][index]
        axes[row_index, 0].imshow(Image.open(reference["sar_path"]).convert("L"), cmap="gray")
        axes[row_index, 0].set_title("SAR input")
        axes[row_index, 1].imshow(Image.open(reference["eo_path"]).convert("RGB"))
        axes[row_index, 1].set_title("EO target")
        for model_index, model in enumerate(models):
            prediction_path = Path(by_model[model][index]["prediction_path"])
            if prediction_path.exists():
                axes[row_index, model_index + 2].imshow(Image.open(prediction_path))
            axes[row_index, model_index + 2].set_title(model.replace("model_", "M").replace("_", " "))
        for axis in axes[row_index]:
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    checkpoints = [Path(path) for path in args.checkpoints] if args.checkpoints else discover_checkpoints(out_dir)
    if not checkpoints:
        raise SystemExit("No checkpoints found. Train at least one model before running evaluation.")
    missing = [str(path) for path in checkpoints if not path.exists()]
    if missing:
        raise SystemExit(f"Missing checkpoints: {missing}")
    dataset = SarOpticalDataset(args.manifest, args.split, args.image_size, False, args.limit)
    loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    eval_root = out_dir / (args.eval_name or f"evaluation_{args.split}")
    eval_root.mkdir(parents=True, exist_ok=True)
    summaries, all_rows = [], []
    for checkpoint in checkpoints:
        summary, rows = evaluate_checkpoint(checkpoint, loader, args, device, eval_root)
        summaries.append(summary)
        all_rows.extend(rows)
    write_csv(summaries, eval_root / "model_comparison.csv")
    write_csv(terrain_summary(all_rows), eval_root / "terrain_comparison.csv")
    (eval_root / "model_comparison.json").write_text(json.dumps(summaries, indent=2, allow_nan=True))
    plot_comparison(summaries, eval_root / "metric_comparison.png")
    if not args.no_save_predictions:
        qualitative_grid(all_rows, eval_root / "qualitative_comparison.png")
    print(json.dumps(summaries, indent=2, allow_nan=True))
    print("Evaluation:", eval_root)


if __name__ == "__main__":
    main()
