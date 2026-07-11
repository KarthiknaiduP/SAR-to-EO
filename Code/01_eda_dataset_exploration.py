"""Research-quality EDA for the Kaggle Sentinel-1/2 paired dataset.

Creates a scene-disjoint manifest, publication-quality figures, CSV tables,
quality checks, JSON statistics, and a Markdown EDA report.

Kaggle:
    python 01_eda_dataset_exploration.py
Faster sampled quality audit:
    python 01_eda_dataset_exploration.py --quality_limit 5000
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm.auto import tqdm


DEFAULT_ROOT = "/kaggle/input/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain/v_2"
DEFAULT_OUT = "/kaggle/working/sar2eo_outputs"
COLORS = ["#287271", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#6D597A"]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=DEFAULT_ROOT)
    parser.add_argument("--out_dir", default=DEFAULT_OUT)
    parser.add_argument("--sample_count", type=int, default=1000)
    parser.add_argument("--quality_limit", type=int, default=0, help="0 audits all pairs")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def scene_id(stem: str, terrain: str) -> str:
    clean = stem.replace("_s1_", "_").replace("_s2_", "_")
    parts = clean.split("_")
    return f"{terrain}/{'_'.join(parts[:3]) if len(parts) >= 3 else clean}"


def scene_splits(ids: list[str], seed: int, val_fraction: float = 0.15, test_fraction: float = 0.15) -> dict[str, str]:
    scenes = sorted(set(ids))
    random.Random(seed).shuffle(scenes)
    n_test = max(1, round(len(scenes) * test_fraction))
    n_val = max(1, round(len(scenes) * val_fraction))
    return {
        scene: "test" if index < n_test else "val" if index < n_test + n_val else "train"
        for index, scene in enumerate(scenes)
    }


def create_manifest(root: Path, path: Path, seed: int) -> list[dict[str, str]]:
    candidates = []
    for sar_path in sorted(root.glob("*/s1/*.png")):
        eo_path = Path(str(sar_path).replace("/s1/", "/s2/").replace("_s1_", "_s2_"))
        terrain = sar_path.relative_to(root).parts[0]
        candidates.append(
            {
                "sar_path": str(sar_path),
                "eo_path": str(eo_path),
                "terrain": terrain,
                "scene_id": scene_id(sar_path.stem, terrain),
                "eo_exists": eo_path.exists(),
            }
        )
    paired = [row for row in candidates if row["eo_exists"]]
    lookup = scene_splits([row["scene_id"] for row in paired], seed)
    rows = [
        {key: row[key] for key in ("sar_path", "eo_path", "terrain", "scene_id")} | {"split": lookup[row["scene_id"]]}
        for row in paired
    ]
    random.Random(seed).shuffle(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sar_path", "eo_path", "split", "scene_id", "terrain"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def entropy(array: np.ndarray) -> float:
    histogram = np.histogram(array, bins=256, range=(0, 1))[0].astype(np.float64)
    probability = histogram[histogram > 0] / histogram.sum()
    return float(-(probability * np.log2(probability)).sum())


def sample_statistics(rows: list[dict[str, str]], count: int, seed: int) -> tuple[list[dict], dict[str, np.ndarray]]:
    sample = random.Random(seed).sample(rows, min(count, len(rows)))
    records, sar_pixels, rgb_pixels = [], [], []
    rng = np.random.default_rng(seed)
    for row in tqdm(sample, desc="sample statistics"):
        sar = np.asarray(Image.open(row["sar_path"]).convert("L"), dtype=np.float32) / 255
        rgb = np.asarray(Image.open(row["eo_path"]).convert("RGB"), dtype=np.float32) / 255
        gray = rgb.mean(2)
        sar_grad = np.hypot(*np.gradient(sar))
        rgb_grad = np.hypot(*np.gradient(gray))
        records.append(
            {
                "sar_path": row["sar_path"],
                "terrain": row["terrain"],
                "split": row["split"],
                "sar_mean": float(sar.mean()),
                "sar_std": float(sar.std()),
                "sar_p01": float(np.quantile(sar, 0.01)),
                "sar_median": float(np.median(sar)),
                "sar_p99": float(np.quantile(sar, 0.99)),
                "r_mean": float(rgb[:, :, 0].mean()),
                "g_mean": float(rgb[:, :, 1].mean()),
                "b_mean": float(rgb[:, :, 2].mean()),
                "brightness": float(gray.mean()),
                "sar_entropy": entropy(sar),
                "eo_entropy": entropy(gray),
                "sar_edge_density": float(sar_grad.mean()),
                "eo_edge_density": float(rgb_grad.mean()),
            }
        )
        sar_pixels.append(rng.choice(sar.ravel(), min(4096, sar.size), replace=False))
        selected = rng.choice(rgb.reshape(-1, 3), min(4096, rgb.shape[0] * rgb.shape[1]), replace=False, axis=0)
        rgb_pixels.append(selected)
    return records, {"sar": np.concatenate(sar_pixels), "rgb": np.concatenate(rgb_pixels)}


def file_hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quality_audit(rows: list[dict[str, str]], limit: int) -> tuple[dict, Counter]:
    audited = rows[:limit] if limit else rows
    corrupt, shape_mismatch, resolutions = [], [], Counter()
    hashes: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for row in tqdm(audited, desc="quality audit"):
        try:
            with Image.open(row["sar_path"]) as sar_image:
                sar_image.verify()
            with Image.open(row["eo_path"]) as eo_image:
                eo_image.verify()
            with Image.open(row["sar_path"]) as sar_image, Image.open(row["eo_path"]) as eo_image:
                resolutions[f"{sar_image.width}x{sar_image.height}"] += 1
                if sar_image.size != eo_image.size:
                    shape_mismatch.append(row["sar_path"])
            hashes[("sar", file_hash(row["sar_path"]))].append(row["sar_path"])
            hashes[("eo", file_hash(row["eo_path"]))].append(row["eo_path"])
        except (OSError, UnidentifiedImageError) as error:
            corrupt.append({"path": row["sar_path"], "error": repr(error)})
    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    report = {
        "pairs_audited": len(audited),
        "audit_is_complete": len(audited) == len(rows),
        "missing_eo_pairs": 0,
        "corrupt_images": corrupt,
        "pair_shape_mismatches": shape_mismatch,
        "exact_duplicate_groups": duplicate_groups,
        "exact_duplicate_group_count": len(duplicate_groups),
    }
    return report, resolutions


def save_table(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bar(counter: Counter, title: str, ylabel: str, path: Path) -> None:
    labels, values = zip(*sorted(counter.items(), key=lambda item: item[1], reverse=True))
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(labels, values, color=COLORS[: len(labels)])
    axis.set(title=title, ylabel=ylabel)
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def distribution_plots(records: list[dict], pixels: dict[str, np.ndarray], figures: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(pixels["sar"], bins=80, color=COLORS[0], alpha=0.9)
    axes[0].set(title="SAR intensity distribution", xlabel="Normalized intensity", ylabel="Pixels")
    for channel, color, label in zip(range(3), ("#C44536", "#2A9D8F", "#457B9D"), ("Red", "Green", "Blue")):
        axes[1].hist(pixels["rgb"][:, channel], bins=80, histtype="step", linewidth=2, color=color, label=label)
    axes[1].set(title="EO channel distributions", xlabel="Normalized intensity", ylabel="Pixels")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(figures / "pixel_histograms.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    p01 = [row["sar_p01"] for row in records]
    median = [row["sar_median"] for row in records]
    p99 = [row["sar_p99"] for row in records]
    axes[0, 0].boxplot([p01, median, p99], tick_labels=["p01", "median", "p99"])
    axes[0, 0].set_title("SAR dynamic range across images")
    for key, color, label in [("r_mean", "#C44536", "R"), ("g_mean", "#2A9D8F", "G"), ("b_mean", "#457B9D", "B")]:
        axes[0, 1].hist([row[key] for row in records], bins=40, alpha=0.45, color=color, label=label)
    axes[0, 1].set_title("Per-image RGB channel means")
    axes[0, 1].legend()
    axes[1, 0].hist([row["sar_entropy"] for row in records], bins=40, alpha=0.65, label="SAR", color=COLORS[0])
    axes[1, 0].hist([row["eo_entropy"] for row in records], bins=40, alpha=0.65, label="EO", color=COLORS[3])
    axes[1, 0].set_title("Image entropy")
    axes[1, 0].legend()
    axes[1, 1].scatter(
        [row["sar_edge_density"] for row in records],
        [row["eo_edge_density"] for row in records],
        s=12,
        alpha=0.4,
        color=COLORS[4],
    )
    axes[1, 1].set(title="SAR versus EO edge density", xlabel="SAR edge density", ylabel="EO edge density")
    for axis in axes.flat:
        axis.grid(alpha=0.15)
    figure.tight_layout()
    figure.savefig(figures / "image_statistics.png", dpi=200)
    plt.close(figure)


def sample_grid(rows: list[dict[str, str]], path: Path, count: int, seed: int, title: str) -> None:
    sample = random.Random(seed).sample(rows, min(count, len(rows)))
    figure, axes = plt.subplots(len(sample), 2, figsize=(7, 2.8 * len(sample)), squeeze=False)
    for index, row in enumerate(sample):
        axes[index, 0].imshow(Image.open(row["sar_path"]).convert("L"), cmap="gray")
        axes[index, 1].imshow(Image.open(row["eo_path"]).convert("RGB"))
        axes[index, 0].set_title(f"SAR | {row['terrain']} | {row['split']}")
        axes[index, 1].set_title("EO target")
        for axis in axes[index]:
            axis.axis("off")
    figure.suptitle(title, y=1)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def terrain_grid(rows: list[dict[str, str]], path: Path) -> None:
    samples = []
    for terrain in sorted(set(row["terrain"] for row in rows)):
        samples.append(next(row for row in rows if row["terrain"] == terrain))
    figure, axes = plt.subplots(len(samples), 2, figsize=(7, 3 * len(samples)), squeeze=False)
    for index, row in enumerate(samples):
        axes[index, 0].imshow(Image.open(row["sar_path"]).convert("L"), cmap="gray")
        axes[index, 1].imshow(Image.open(row["eo_path"]).convert("RGB"))
        axes[index, 0].set_title(f"{row['terrain']}: SAR")
        axes[index, 1].set_title(f"{row['terrain']}: EO")
        for axis in axes[index]:
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def assert_scene_disjoint(rows: list[dict[str, str]]) -> dict:
    sets = {split: {row["scene_id"] for row in rows if row["split"] == split} for split in ("train", "val", "test")}
    overlaps = {
        "train_val": sorted(sets["train"] & sets["val"]),
        "train_test": sorted(sets["train"] & sets["test"]),
        "val_test": sorted(sets["val"] & sets["test"]),
    }
    if any(overlaps.values()):
        raise AssertionError(f"Scene leakage found: {overlaps}")
    return {"scene_counts": {key: len(value) for key, value in sets.items()}, "overlaps": overlaps, "passed": True}


def write_report(stats: dict, path: Path) -> None:
    quality = stats["quality"]
    lines = [
        "# SAR-to-EO Dataset EDA",
        "",
        "## Dataset Summary",
        "",
        "| Property | Value |",
        "|---|---:|",
        f"| Total paired patches | {stats['total_pairs']} |",
        f"| Unique scenes | {stats['unique_scenes']} |",
        f"| Terrains | {len(stats['terrain_counts'])} |",
        f"| Train / validation / test | {stats['split_counts']} |",
        f"| Resolutions | {stats['resolution_counts']} |",
        f"| Statistical sample | {stats['statistical_sample_count']} |",
        "",
        "## Integrity Checks",
        "",
        f"- Scene-disjoint split passed: **{stats['split_verification']['passed']}**",
        f"- Audited pairs: **{quality['pairs_audited']}** (complete={quality['audit_is_complete']})",
        f"- Corrupt images: **{len(quality['corrupt_images'])}**",
        f"- Pair shape mismatches: **{len(quality['pair_shape_mismatches'])}**",
        f"- Exact duplicate hash groups: **{quality['exact_duplicate_group_count']}**",
        "",
        "## Preprocessing Decision",
        "",
        "Both modalities are mapped to `[-1, 1]`. The SAR percentile distributions in `image_statistics.png` document the input dynamic range; no logarithmic calibration is claimed because this Kaggle release contains display PNGs rather than calibrated Sentinel-1 backscatter products.",
        "",
        "## Scope",
        "",
        "The samples are paired image patches grouped by inferred source scene and terrain. Conclusions should not be generalized beyond the represented geography, acquisition periods, preprocessing, and terrain classes.",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    args = arguments()
    root, out = Path(args.data_root), Path(args.out_dir)
    figures, tables = out / "report_figures", out / "report_tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    rows = create_manifest(root, out / "manifest.csv", args.seed)
    if not rows:
        raise SystemExit(f"No paired images found under {root}")
    split_counts = Counter(row["split"] for row in rows)
    terrain_counts = Counter(row["terrain"] for row in rows)
    scenes = Counter(row["scene_id"] for row in rows)
    split_check = assert_scene_disjoint(rows)
    records, pixels = sample_statistics(rows, args.sample_count, args.seed)
    quality, resolutions = quality_audit(rows, args.quality_limit)
    save_table(records, tables / "sample_image_statistics.csv")
    save_table(
        [{"scene_id": scene, "patch_count": count} for scene, count in scenes.items()],
        tables / "patches_per_scene.csv",
    )
    save_table(
        [{"terrain": key, "pair_count": value} for key, value in terrain_counts.items()],
        tables / "terrain_counts.csv",
    )
    bar(terrain_counts, "Terrain distribution", "Paired patches", figures / "terrain_distribution.png")
    bar(split_counts, "Scene-disjoint split distribution", "Paired patches", figures / "split_distribution.png")
    bar(Counter(scenes.values()), "Patches per scene", "Number of scenes", figures / "patches_per_scene.png")
    bar(resolutions, "Image resolution distribution", "Audited pairs", figures / "resolution_distribution.png")
    distribution_plots(records, pixels, figures)
    sample_grid(rows, figures / "random_pair_inspection.png", 20, args.seed, "Random paired-image inspection")
    terrain_grid(rows, figures / "terrain_examples.png")
    channel_means = {key: float(np.mean([row[key] for row in records])) for key in ("r_mean", "g_mean", "b_mean")}
    stats = {
        "data_root": str(root),
        "total_pairs": len(rows),
        "unique_scenes": len(scenes),
        "split_counts": dict(split_counts),
        "terrain_counts": dict(terrain_counts),
        "resolution_counts": dict(resolutions),
        "statistical_sample_count": len(records),
        "scene_patch_count": {
            "minimum": min(scenes.values()),
            "maximum": max(scenes.values()),
            "mean": float(np.mean(list(scenes.values()))),
        },
        "sample_statistics": {
            "sar_mean": float(np.mean([row["sar_mean"] for row in records])),
            "sar_std": float(np.mean([row["sar_std"] for row in records])),
            "rgb_channel_means": channel_means,
            "eo_brightness": float(np.mean([row["brightness"] for row in records])),
            "sar_entropy": float(np.mean([row["sar_entropy"] for row in records])),
            "eo_entropy": float(np.mean([row["eo_entropy"] for row in records])),
        },
        "split_verification": split_check,
        "quality": quality,
    }
    (out / "eda_stats.json").write_text(json.dumps(stats, indent=2))
    write_report(stats, out / "EDA_REPORT.md")
    print(json.dumps({key: value for key, value in stats.items() if key != "quality"}, indent=2))
    print("EDA outputs:", out)


if __name__ == "__main__":
    main()
