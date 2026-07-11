# SAR-to-EO Dataset EDA

## Dataset Summary

| Property | Value |
|---|---:|
| Total paired patches | 16000 |
| Unique scenes | 32 |
| Terrains | 4 |
| Train / validation / test | {'train': 10775, 'test': 2711, 'val': 2514} |
| Resolutions | {'256x256': 16000} |
| Statistical sample | 1000 |

## Integrity Checks

- Scene-disjoint split passed: **True**
- Audited pairs: **16000** (complete=True)
- Corrupt images: **0**
- Pair shape mismatches: **0**
- Exact duplicate hash groups: **2**

## Preprocessing Decision

Both modalities are mapped to `[-1, 1]`. The SAR percentile distributions in `image_statistics.png` document the input dynamic range; no logarithmic calibration is claimed because this Kaggle release contains display PNGs rather than calibrated Sentinel-1 backscatter products.

## Scope

The samples are paired image patches grouped by inferred source scene and terrain. Conclusions should not be generalized beyond the represented geography, acquisition periods, preprocessing, and terrain classes.