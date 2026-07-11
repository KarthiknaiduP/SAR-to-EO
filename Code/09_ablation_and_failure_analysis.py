"""Controlled ablation and evidence-based failure analysis.

By default this analyzes outputs already produced by 08. Add --train_ablations
to train controlled variants of the final paired model (no attention, no
perceptual loss, no feature matching, no gradient consistency) and evaluate
them against the full final model.

Kaggle:
    python 09_ablation_and_failure_analysis.py
Full ablation:
    python 09_ablation_and_failure_analysis.py --train_ablations --epochs 20
Smoke test:
    python 09_ablation_and_failure_analysis.py --train_ablations --quick
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from sar2eo_common import DEFAULT_MANIFEST, DEFAULT_OUT


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out_dir", default=DEFAULT_OUT)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--train_ablations", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train_limit", type=int, default=8000)
    parser.add_argument("--val_limit", type=int, default=1000)
    parser.add_argument("--test_limit", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--worst_count", type=int, default=8)
    parser.add_argument("--include_fid", action="store_true")
    parser.add_argument("--include_lpips", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("\nRUN:", " ".join(command))
    subprocess.run(command, check=True)


def train_ablations(args: argparse.Namespace, here: Path) -> Path:
    if args.quick:
        args.epochs, args.train_limit, args.val_limit, args.test_limit, args.batch_size = 1, 64, 16, 16, 2
    specifications = [
        ("ablation_final_full", []),
        ("ablation_final_no_attention", ["--no_attention"]),
        ("ablation_final_no_perceptual", ["--lambda_perceptual", "0"]),
        ("ablation_final_no_feature_matching", ["--lambda_feature_matching", "0"]),
        ("ablation_final_no_gradient", ["--lambda_gradient", "0"]),
    ]
    checkpoints = []
    for run_name, extras in specifications:
        checkpoint = Path(args.out_dir) / run_name / "checkpoints" / "best.pt"
        checkpoints.append(checkpoint)
        command = [
            sys.executable,
            str(here / "07_train_final_fusion_gan.py"),
            "--manifest",
            args.manifest,
            "--out_dir",
            args.out_dir,
            "--run_name",
            run_name,
            "--epochs",
            str(args.epochs),
            "--train_limit",
            str(args.train_limit),
            "--val_limit",
            str(args.val_limit),
            "--batch_size",
            str(args.batch_size),
            "--num_workers",
            str(args.num_workers),
            "--warm_start",
            "none",
        ] + extras
        run(command)
    evaluation = [
        sys.executable,
        str(here / "08_evaluate_all_models.py"),
        "--manifest",
        args.manifest,
        "--out_dir",
        args.out_dir,
        "--split",
        args.split,
        "--eval_name",
        f"evaluation_{args.split}_final_ablations",
        "--limit",
        str(args.test_limit),
        "--checkpoints",
        *map(str, checkpoints),
    ]
    if not args.include_fid:
        evaluation.append("--skip_fid")
    if not args.include_lpips:
        evaluation.append("--skip_lpips")
    run(evaluation)
    return Path(args.out_dir) / f"evaluation_{args.split}_final_ablations"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def numeric(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def image_attributes(row: dict[str, str]) -> dict[str, float]:
    sar = np.asarray(Image.open(row["sar_path"]).convert("L"), dtype=np.float32) / 255
    eo = np.asarray(Image.open(row["eo_path"]).convert("RGB"), dtype=np.float32) / 255
    gray = eo.mean(axis=2)
    grad_y, grad_x = np.gradient(gray)
    return {
        "sar_mean": float(sar.mean()),
        "sar_std": float(sar.std()),
        "eo_brightness": float(gray.mean()),
        "eo_contrast": float(gray.std()),
        "eo_edge_density": float(np.mean(np.hypot(grad_x, grad_y))),
    }


def correlation(x: list[float], y: list[float]) -> float:
    x_array, y_array = np.asarray(x), np.asarray(y)
    valid = np.isfinite(x_array) & np.isfinite(y_array)
    if valid.sum() < 3 or x_array[valid].std() == 0 or y_array[valid].std() == 0:
        return float("nan")
    return float(np.corrcoef(x_array[valid], y_array[valid])[0, 1])


def analyze_model(metrics_file: Path, analysis_dir: Path, worst_count: int) -> dict:
    rows = read_csv(metrics_file)
    model = rows[0]["model"]
    attributes = []
    for row in rows:
        values = image_attributes(row)
        row.update({key: str(value) for key, value in values.items()})
        attributes.append(values)
    worst = sorted(rows, key=lambda row: numeric(row, "ssim"))[:worst_count]
    figure, axes = plt.subplots(len(worst), 4, figsize=(12, 3 * len(worst)), squeeze=False)
    for index, row in enumerate(worst):
        sar = np.asarray(Image.open(row["sar_path"]).convert("L"))
        target = np.asarray(Image.open(row["eo_path"]).convert("RGB").resize((256, 256)), dtype=np.float32) / 255
        prediction_path = Path(row["prediction_path"])
        prediction = np.asarray(Image.open(prediction_path).convert("RGB"), dtype=np.float32) / 255
        error = np.mean(np.abs(prediction - target), axis=2)
        axes[index, 0].imshow(sar, cmap="gray")
        axes[index, 1].imshow(target)
        axes[index, 2].imshow(prediction)
        axes[index, 3].imshow(error, cmap="magma", vmin=0, vmax=max(0.5, float(error.max())))
        titles = ("SAR", "Target", f"Prediction\nSSIM={numeric(row, 'ssim'):.3f}", "Absolute error")
        for axis, title in zip(axes[index], titles):
            axis.set_title(title)
            axis.axis("off")
    figure.suptitle(f"Worst test cases: {model}", y=1.0)
    figure.tight_layout()
    model_dir = analysis_dir / model
    model_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(model_dir / "worst_cases.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    by_terrain: defaultdict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_terrain[row["terrain"]].append(row)
    terrain_rows = []
    for terrain, group in sorted(by_terrain.items()):
        terrain_rows.append(
            {
                "terrain": terrain,
                "count": len(group),
                "mean_ssim": float(np.mean([numeric(row, "ssim") for row in group])),
                "mean_psnr": float(np.mean([numeric(row, "psnr") for row in group])),
                "mean_mae": float(np.mean([numeric(row, "mae") for row in group])),
            }
        )
    with (model_dir / "terrain_failures.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(terrain_rows[0]))
        writer.writeheader()
        writer.writerows(terrain_rows)
    correlations = {}
    for attribute in attributes[0]:
        correlations[f"{attribute}_vs_ssim"] = correlation(
            [float(row[attribute]) for row in rows], [numeric(row, "ssim") for row in rows]
        )
        correlations[f"{attribute}_vs_mae"] = correlation(
            [float(row[attribute]) for row in rows], [numeric(row, "mae") for row in rows]
        )
    report = {
        "model": model,
        "worst_terrain_by_ssim": min(terrain_rows, key=lambda row: row["mean_ssim"])["terrain"],
        "correlations": correlations,
        "worst_examples": [
            {
                "scene_id": row["scene_id"],
                "terrain": row["terrain"],
                "ssim": numeric(row, "ssim"),
                "psnr": numeric(row, "psnr"),
            }
            for row in worst
        ],
    }
    (model_dir / "failure_summary.json").write_text(json.dumps(report, indent=2, allow_nan=True))
    return report


def write_interpretation(reports: list[dict], output: Path) -> None:
    lines = [
        "# Ablation and Failure Analysis",
        "",
        "The statements below describe measured associations, not causal explanations.",
        "",
    ]
    for report in reports:
        finite = [(key, value) for key, value in report["correlations"].items() if math.isfinite(value)]
        strongest = max(finite, key=lambda item: abs(item[1])) if finite else ("unavailable", float("nan"))
        lines += [
            f"## {report['model']}",
            "",
            f"- Lowest-SSIM terrain: **{report['worst_terrain_by_ssim']}**.",
            f"- Strongest measured image-property association: `{strongest[0]}` = {strongest[1]:.3f}.",
            "- Inspect `worst_cases.png` before assigning a physical cause; SAR-to-EO translation is underdetermined.",
            "",
        ]
    lines += [
        "## Interpretation Rules",
        "",
        "- A component is considered useful only if it improves repeated, same-seed controlled runs on the fixed test split.",
        "- PSNR/SSIM reward fidelity; LPIPS/FID reward perceptual similarity. Report disagreements rather than selecting only a favorable metric.",
        "- Generated colors and objects are hypotheses, not observations. Failure analysis must explicitly discuss hallucination risk.",
    ]
    output.write_text("\n".join(lines))


def main() -> None:
    args = arguments()
    here = Path(__file__).resolve().parent
    evaluation_dir = (
        train_ablations(args, here)
        if args.train_ablations
        else Path(args.out_dir) / f"evaluation_{args.split}"
    )
    metric_files = sorted(evaluation_dir.glob("*/per_image_metrics.csv"))
    if not metric_files:
        raise SystemExit(f"No per-image metrics under {evaluation_dir}. Run 08_evaluate_all_models.py first.")
    analysis_dir = Path(args.out_dir) / "ablation_failure_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    reports = [analyze_model(path, analysis_dir, args.worst_count) for path in metric_files]
    (analysis_dir / "failure_summaries.json").write_text(json.dumps(reports, indent=2, allow_nan=True))
    comparison = evaluation_dir / "model_comparison.csv"
    if args.train_ablations and comparison.exists():
        (analysis_dir / "ablation_comparison.csv").write_text(comparison.read_text())
    write_interpretation(reports, analysis_dir / "FAILURE_ANALYSIS.md")
    print("Analysis:", analysis_dir)


if __name__ == "__main__":
    main()
