# SAR-to-EO Image Translation Using Progressive Generative Models

This repository contains a research-style SAR-to-EO image translation project developed for the GalaxEye Satellite AI Research Intern technical assignment.

The project studies paired Sentinel-1 SAR and Sentinel-2 optical image translation through a progressive sequence of generative models rather than a single fixed architecture.

## What is included

```text
code/                         Training, evaluation, EDA, ablation, and report scripts
figures/                      EDA, training, evaluation, ablation, and failure-analysis figures
outputs/                      CSV/JSON/Markdown result artifacts
reports/                      Final PDF and LaTeX report
checkpoints/                  Checkpoint notes and external-storage guidance
docs/                         Run-order documentation
infer.py                      Assignment-style inference entry point
sar2eo_common.py              Shared model, data, and training utilities for inference
requirements.txt              Kaggle/Python dependency list
```

## Model progression

| Model | Purpose |
|---|---|
| M1: U-Net L1 | Conservative supervised reconstruction baseline |
| M2: Pix2Pix | Adversarial paired translation baseline |
| M3: Pix2Pix + perceptual/FM | Feature-space and discriminator feature-matching supervision |
| M4: Attention ResUNet | Structure-aware generator with spectral PatchGAN |
| M5: Final Fusion GAN | Final paired GAN with gradient consistency and EMA |

## Final evaluation summary

Evaluation was performed on 200-sample validation and test subsets with PSNR, SSIM, MAE, LPIPS, FID, and runtime.

On the held-out test subset:

| Best criterion | Winning model |
|---|---|
| PSNR | M4 Attention ResUNet |
| SSIM | M1 U-Net L1 |
| MAE | M5 Final Fusion GAN |
| LPIPS | M2 Pix2Pix |
| FID | M3 Pix2Pix + perceptual/FM |

The core conclusion is that no single model dominates all metrics. Pixel fidelity, perceptual similarity, and distribution-level realism reward different behaviours.

## Dataset

The dataset is not committed to GitHub.

Expected dataset structure:

```text
v_2/
  agri/
    s1/
    s2/
  urban/
    s1/
    s2/
  barrenland/
    s1/
    s2/
  grassland/
    s1/
    s2/
```

The project was run with the Kaggle dataset:

```text
Sentinel-1&2 Image Pairs (SAR & Optical)
requiemonk/sentinel12-image-pairs-segregated-by-terrain
```

## Run order

The intended run order is:

```bash
python code/01_eda_dataset_exploration.py
python code/02_dataset_dataloader.py
python code/03_train_unet_l1.py
python code/04_train_pix2pix.py
python code/05_train_pix2pix_perceptual.py
python code/06_train_attention_resunet.py
python code/07_train_final_fusion_gan.py
python code/08_evaluate_all_models.py
python code/09_ablation_and_failure_analysis.py
python code/10_generate_report_results.py
```

See `docs/README_RUN_ORDER.md` for more detailed Kaggle-oriented instructions.

## Inference

After placing a trained checkpoint in `checkpoints/`, run:

```bash
python infer.py \
  --checkpoint checkpoints/model_5_final_fusion_gan_best.pt \
  --input path/to/sar.png \
  --output generated_eo.png
```

The checkpoint files are not committed because each model checkpoint is over GitHub's normal 100 MB file limit. See `checkpoints/README.md`.

## Main report

Final report:

```text
reports/report_final.pdf
```

Source LaTeX:

```text
reports/report_final.tex
```

## Important result artifacts

```text
outputs/evaluation/test_with_fid/model_comparison.csv
outputs/evaluation/val_with_fid/model_comparison.csv
outputs/ablation/model_comparison.csv
outputs/failure_analysis/FAILURE_ANALYSIS.md
figures/evaluation/metric_comparison_test_fid.png
figures/evaluation/metric_comparison_val_fid.png
figures/evaluation/qualitative_comparison_test_fid.png
figures/evaluation/ablation_metric_comparison.png
```

## Notes

This project prioritizes reproducible experimentation, data understanding, architecture comparison, ablation, and failure analysis. The final recommendation is not that one GAN solves SAR-to-EO translation, but that SAR-to-EO generation needs explicit structure preservation and uncertainty-aware modelling, with conditional diffusion as a strong future direction.
