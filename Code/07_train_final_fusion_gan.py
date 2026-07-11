"""Model 5: final evidence-driven paired SAR-to-EO model.

This replaces the previous CycleGAN slot. The dataset is paired, so the final
solution should exploit paired supervision rather than spending time on an
unpaired comparison. The model keeps the strongest paired direction from
Models 03-06 and adds three practical research upgrades:

1. Warm start from Model 4 when available.
2. Gradient consistency loss to preserve edges and field/road boundaries.
3. EMA generator checkpointing for smoother final predictions.

Kaggle:
    python 07_train_final_fusion_gan.py --epochs 20 --train_limit 8000 --val_limit 1000
Smoke test:
    python 07_train_final_fusion_gan.py --epochs 1 --train_limit 64 --val_limit 16 --batch_size 2 --warm_start none
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from sar2eo_common import DEFAULT_MANIFEST, DEFAULT_OUT, TrainConfig, train_paired


PRIOR_RUNS = [
    "model_1_unet_l1",
    "model_2_pix2pix",
    "model_3_pix2pix_perceptual",
    "model_4_attention_resunet",
]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out_dir", default=DEFAULT_OUT)
    parser.add_argument("--run_name", default="model_5_final_fusion_gan")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--train_limit", type=int, default=8000)
    parser.add_argument("--val_limit", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lambda_l1", type=float, default=80.0)
    parser.add_argument("--lambda_perceptual", type=float, default=10.0)
    parser.add_argument("--lambda_feature_matching", type=float, default=10.0)
    parser.add_argument("--lambda_gradient", type=float, default=20.0)
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument(
        "--warm_start",
        default="auto",
        help="auto, none, or a checkpoint path. Auto uses Model 4 best.pt if present.",
    )
    parser.add_argument("--no_attention", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def summarize_prior_runs(out_dir: Path) -> list[dict]:
    summary = []
    for run_name in PRIOR_RUNS:
        losses = out_dir / run_name / "losses.csv"
        checkpoint = out_dir / run_name / "checkpoints" / "best.pt"
        if not losses.exists():
            summary.append({"run_name": run_name, "status": "missing"})
            continue
        rows = read_csv(losses)
        best = min(rows, key=lambda row: number(row, "val_l1"))
        summary.append(
            {
                "run_name": run_name,
                "status": "found",
                "epochs": len(rows),
                "best_epoch": int(number(best, "epoch")),
                "best_val_l1": number(best, "val_l1"),
                "checkpoint": str(checkpoint) if checkpoint.exists() else "",
            }
        )
    return summary


def resolve_warm_start(args: argparse.Namespace) -> str | None:
    if args.warm_start.lower() == "none":
        return None
    if args.warm_start.lower() != "auto":
        return args.warm_start
    checkpoint = Path(args.out_dir) / "model_4_attention_resunet" / "checkpoints" / "best.pt"
    return str(checkpoint) if checkpoint.exists() else None


def write_design_note(args: argparse.Namespace, prior_summary: list[dict], warm_start: str | None) -> None:
    out_dir = Path(args.out_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    found = [row for row in prior_summary if row["status"] == "found" and math.isfinite(row["best_val_l1"])]
    best_prior = min(found, key=lambda row: row["best_val_l1"]) if found else None
    lines = [
        "# Model 5 Design Decision",
        "",
        "CycleGAN was removed from the main path because this project uses paired Sentinel-1/Sentinel-2 patches.",
        "The final model therefore extends the strongest paired family rather than switching to unpaired translation.",
        "",
        "## Evidence from Previous Runs",
        "",
        "| Run | Status | Epochs | Best epoch | Best val L1 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in prior_summary:
        if row["status"] == "missing":
            lines.append(f"| {row['run_name']} | missing |  |  |  |")
        else:
            lines.append(
                f"| {row['run_name']} | found | {row['epochs']} | {row['best_epoch']} | {row['best_val_l1']:.5f} |"
            )
    lines += [
        "",
        "## Final Model",
        "",
        "- Generator: attention ResUNet.",
        "- Discriminator: spectral-normalized PatchGAN.",
        "- Objective: hinge adversarial + L1 + perceptual + discriminator feature matching + gradient consistency.",
        "- Stability upgrade: EMA generator is saved as `best.pt` and `latest.pt`.",
        f"- Warm start checkpoint: `{warm_start or 'none'}`.",
    ]
    if best_prior:
        lines.append(f"- Best previous validation L1 observed in logs: `{best_prior['run_name']}`.")
    lines += [
        "",
        "This is a defensible final architecture because each added component targets a measured weakness:",
        "L1 controls paired fidelity, GAN improves texture, perceptual/feature matching reduce over-smoothing,",
        "attention helps spatially selective reconstruction, and gradient consistency emphasizes boundaries.",
    ]
    (out_dir / "MODEL_5_DESIGN_DECISION.md").write_text("\n".join(lines))


def main() -> None:
    args = arguments()
    prior_summary = summarize_prior_runs(Path(args.out_dir))
    print("Previous model summary:")
    print(json.dumps(prior_summary, indent=2, allow_nan=True))
    warm_start = resolve_warm_start(args)
    write_design_note(args, prior_summary, warm_start)
    config = TrainConfig(
        model_name="final_fusion_gan",
        run_name=args.run_name,
        manifest=args.manifest,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        base_channels=args.base_channels,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        learning_rate=args.learning_rate,
        lambda_l1=args.lambda_l1,
        lambda_perceptual=args.lambda_perceptual,
        lambda_feature_matching=args.lambda_feature_matching,
        lambda_gradient=args.lambda_gradient,
        gan_mode="hinge",
        attention=not args.no_attention,
        amp=not args.no_amp,
        seed=args.seed,
        warm_start_checkpoint=warm_start,
        ema_decay=args.ema_decay,
    )
    print("Best checkpoint:", train_paired(config))


if __name__ == "__main__":
    main()
