# Kaggle SAR-to-EO: Exact Run Order

The dataset input is read-only. Upload the updated `kaggle_notebook_scripts.zip`
as a Kaggle Dataset, attach it to the notebook, and copy it to working storage.

```python
!find /kaggle/input -name kaggle_notebook_scripts.zip
```

Use the path printed above:

```python
!unzip -q -o "/kaggle/input/YOUR-UPLOAD-PATH/kaggle_notebook_scripts.zip" -d /kaggle/working
%cd /kaggle/working/kaggle_notebook_scripts
```

Enable a GPU in **Settings > Accelerator > GPU**.

## 0. Dependencies

```python
!pip install -q lpips scikit-image "torchmetrics[image]" torch-fidelity
```

PyTorch, torchvision, NumPy, Pillow, Matplotlib, and tqdm are normally
preinstalled by Kaggle.

## 1. EDA and Manifest

```python
!python 01_eda_dataset_exploration.py
```

This creates the scene-disjoint manifest used by every experiment:

```text
/kaggle/working/sar2eo_outputs/manifest.csv
```

## 2. DataLoader Validation

```python
!python 02_dataset_dataloader.py
```

Do not train until the printed SAR/EO shapes and preview are correct.

## 3. One-Minute Pipeline Test

Run one tiny model first:

```python
!python 03_train_unet_l1.py --run_name smoke_test --epochs 1 --train_limit 64 --val_limit 16 --batch_size 2
```

## 4. Progressive Model Experiments

For an initial short-period study, use the same `15 epochs / 4000 train
samples / seed 42` budget for Models 1-4.

```python
!python 03_train_unet_l1.py --epochs 15 --train_limit 4000 --val_limit 500 --batch_size 4 --seed 42
```

```python
!python 04_train_pix2pix.py --epochs 15 --train_limit 4000 --val_limit 500 --batch_size 4 --seed 42
```

```python
!python 05_train_pix2pix_perceptual.py --epochs 15 --train_limit 4000 --val_limit 500 --batch_size 4 --seed 42
```

```python
!python 06_train_attention_resunet.py --epochs 15 --train_limit 4000 --val_limit 500 --batch_size 4 --seed 42
```

After 03-06 finish, train the final evidence-driven model. It reads the logs
from Models 1-4, warm-starts from Model 4 if that checkpoint exists, and writes
`MODEL_5_DESIGN_DECISION.md`.

```python
!python 07_train_final_fusion_gan.py --epochs 20 --train_limit 8000 --val_limit 1000 --batch_size 4 --seed 42
```

For final stronger runs, increase Models 1-4 to `--epochs 30 --train_limit
8000 --val_limit 1000`. Compare models trained with the same budget; do not
quietly give the preferred model more data or epochs. Model 5 may use fewer
epochs because it is warm-started from Model 4; report that explicitly.

## 5. Common Evaluation

Fast evaluation:

```python
!python 08_evaluate_all_models.py --limit 200 --skip_fid
```

Final evaluation:

```python
!python 08_evaluate_all_models.py --limit 1000
```

Results are saved under:

```text
/kaggle/working/sar2eo_outputs/evaluation_test/
```

## 6. Failure Analysis

Analyze the five main experiments:

```python
!python 09_ablation_and_failure_analysis.py
```

Train controlled final-model ablations only after the main experiments are
working:

```python
!python 09_ablation_and_failure_analysis.py --train_ablations --epochs 15 --train_limit 4000 --val_limit 500 --test_limit 500
```

## 7. Final Report Artifacts

```python
!python 10_generate_report_results.py
```

The final tables, figures, LaTeX, and Markdown report are written to:

```text
/kaggle/working/sar2eo_outputs/final_report/
```

## 8. Download Everything

```python
!cd /kaggle/working && zip -qr sar2eo_complete_results.zip sar2eo_outputs
```

Refresh the Kaggle **Output** panel and download
`sar2eo_complete_results.zip`. Use **Save Version** as an additional backup
before the session ends.
