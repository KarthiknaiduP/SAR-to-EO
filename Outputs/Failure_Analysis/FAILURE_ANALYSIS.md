# Ablation and Failure Analysis

The statements below describe measured associations, not causal explanations.

## model_1_unet_l1

- Lowest-SSIM terrain: **urban**.
- Strongest measured image-property association: `eo_edge_density_vs_ssim` = -0.854.
- Inspect `worst_cases.png` before assigning a physical cause; SAR-to-EO translation is underdetermined.

## model_2_pix2pix

- Lowest-SSIM terrain: **urban**.
- Strongest measured image-property association: `eo_edge_density_vs_ssim` = -0.792.
- Inspect `worst_cases.png` before assigning a physical cause; SAR-to-EO translation is underdetermined.

## model_3_pix2pix_perceptual

- Lowest-SSIM terrain: **urban**.
- Strongest measured image-property association: `eo_edge_density_vs_ssim` = -0.783.
- Inspect `worst_cases.png` before assigning a physical cause; SAR-to-EO translation is underdetermined.

## model_4_attention_resunet

- Lowest-SSIM terrain: **urban**.
- Strongest measured image-property association: `eo_edge_density_vs_ssim` = -0.872.
- Inspect `worst_cases.png` before assigning a physical cause; SAR-to-EO translation is underdetermined.

## model_5_final_fusion_gan

- Lowest-SSIM terrain: **urban**.
- Strongest measured image-property association: `eo_edge_density_vs_ssim` = -0.879.
- Inspect `worst_cases.png` before assigning a physical cause; SAR-to-EO translation is underdetermined.

## Interpretation Rules

- A component is considered useful only if it improves repeated, same-seed controlled runs on the fixed test split.
- PSNR/SSIM reward fidelity; LPIPS/FID reward perceptual similarity. Report disagreements rather than selecting only a favorable metric.
- Generated colors and objects are hypotheses, not observations. Failure analysis must explicitly discuss hallucination risk.