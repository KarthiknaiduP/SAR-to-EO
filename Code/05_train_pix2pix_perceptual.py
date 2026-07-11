"""Model 3: Pix2Pix with perceptual and discriminator feature-matching losses.

Research question: do feature-space constraints improve structure and texture
without changing the Model 2 generator architecture?

Kaggle:
    python 05_train_pix2pix_perceptual.py --epochs 30 --train_limit 8000
"""

import argparse

from sar2eo_common import DEFAULT_MANIFEST, DEFAULT_OUT, TrainConfig, train_paired


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out_dir", default=DEFAULT_OUT)
    parser.add_argument("--run_name", default="model_3_pix2pix_perceptual")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--train_limit", type=int, default=8000)
    parser.add_argument("--val_limit", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lambda_l1", type=float, default=100.0)
    parser.add_argument("--lambda_perceptual", type=float, default=10.0)
    parser.add_argument("--lambda_feature_matching", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    config = TrainConfig(
        model_name="pix2pix_perceptual",
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
        gan_mode="bce",
        amp=not args.no_amp,
        seed=args.seed,
    )
    print("Best checkpoint:", train_paired(config))


if __name__ == "__main__":
    main()
