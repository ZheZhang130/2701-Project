# Advanced-model experiment results

Development selection estimates; not independent final results. The current Stage 3 deployment bundle remains unchanged.

## Original 16/5 driver split

| Candidate | Accuracy | Macro F1 | Log loss | Recall at <=5% safe FPR (same-validation diagnostic) |
|---|---:|---:|---:|---:|
| resnet50_320 | 85.9% | 0.858 | 0.522 | 85.3% |
| convnext_tiny320 | 90.0% | 0.898 | 0.370 | 93.2% |
| regions_resnet18_224 | 80.7% | 0.812 | 0.776 | 77.1% |
| dino_small_frozen224 | 69.1% | 0.685 | 0.967 | 31.6% |
| dino_small_last2_224 | 81.0% | 0.812 | 0.755 | 71.1% |
| existing_ensemble | 84.3% | 0.840 | 0.500 | 93.6% |
| finalist_ensemble | 90.1% | 0.899 | 0.333 | 88.4% |

## Targeted error follow-up

These are same-validation diagnostics with a candidate-specific pooled <=5% FPR threshold; no temporal smoothing.

| Candidate | Safe classified as passenger (of 406) | p072 phone-right recall | p072 phone-left recall | p081 safe FPR at pooled threshold |
|---|---:|---:|---:|---:|
| resnet50_320 | 74 | 100.0% | 70.6% | 17.0% |
| convnext_tiny320 | 101 | 91.7% | 85.3% | 9.0% |
| regions_resnet18_224 | 48 | 25.0% | 0.0% | 0.0% |
| dino_small_frozen224 | 44 | 0.0% | 5.9% | 0.0% |
| dino_small_last2_224 | 102 | 13.9% | 29.4% | 0.0% |
| existing_ensemble | 82 | 2.8% | 0.0% | 15.0% |
| finalist_ensemble | 90 | 100.0% | 79.4% | 17.0% |

## Finalist second seeds

| Candidate | Seed | Accuracy | Log loss | Recall at <=5% safe FPR (diagnostic) |
|---|---:|---:|---:|---:|
| convnext_tiny320 | 3702 | 89.5% | 0.401 | 95.3% |
| resnet50_320 | 3702 | 90.3% | 0.460 | 93.5% |

## Driver-grouped confirmation

Fit11/calibrate3/evaluate7 per fold. Thresholds transfer unchanged from calibration drivers. These models differ from original-split screen bundles.

| Candidate | Pooled safe FPR | Pooled non-safe recall | Worst driver FPR | Lowest driver recall | Development promotion gate |
|---|---:|---:|---:|---:|---|
| original_resnet18_320 | 9.2% | 77.7% | 64.6% | 52.7% | baseline |
| convnext_tiny320 | 18.0% | 92.2% | 98.8% | 51.9% | False |
| resnet50_320 | 10.9% | 89.0% | 92.6% | 65.6% | False |
| existing_ensemble | 4.8% | 84.0% | 100.0% | 61.6% | False |
| finalist_ensemble | 14.6% | 92.4% | 100.0% | 63.3% | False |

## Existing raw-threshold diagnostic

Post-hoc diagnostic using the pre-existing 0.97 raw threshold; no temporal smoothing, no threshold search, not the promotion gate.

| Candidate | Safe FPR | Non-safe recall | Worst driver FPR |
|---|---:|---:|---:|
| original_resnet18_320 | 11.6% | 86.3% | 63.1% |
| convnext_tiny320 | 16.3% | 92.4% | 96.4% |
| resnet50_320 | 15.5% | 90.0% | 95.0% |
| existing_ensemble | 4.1% | 86.2% | 44.4% |
| finalist_ensemble | 9.2% | 89.3% | 85.2% |

## Paired uncertainty

Bootstrap resamples 21 drivers (5,000 draws). Intervals describe paired pooled-rate differences from baseline, not unbiased final confidence; selected models share training groups and development selection.

| Candidate | FPR difference, 95% interval (percentage points) | Recall difference, 95% interval (percentage points) |
|---|---:|---:|
| convnext_tiny320 | -0.3 to 20.1 | 7.9 to 21.1 |
| resnet50_320 | -8.9 to 14.1 | 5.7 to 17.5 |
| existing_ensemble | -13.1 to 5.6 | 1.5 to 11.1 |
| finalist_ensemble | -4.1 to 17.1 | 8.8 to 20.7 |

## Export and runtime verification

| Candidate | MPS median / p95 ms | Export max probability difference | CPU/MPS max difference |
|---|---:|---:|---:|
| convnext_tiny320 | 14.28 / 15.19 | 0 | 1.06e-06 |
| resnet50_320 | 11.68 / 12.23 | 0 | 1.18e-06 |
| existing_ensemble | 17.45 / 25.95 | 5.55e-16 | 8.35e-07 |
| finalist_ensemble | 25.79 / 26.44 | 5.55e-16 | 1.12e-06 |

Timing includes preprocessing and inference, excluding disk/network. See verification JSON for full monitoring smoke scope.

| Candidate / scenario | Accuracy | Safe FPR at fixed clean diagnostic threshold | Non-safe recall |
|---|---:|---:|---:|
| convnext_tiny320 / clean | 90.0% | 4.4% | 93.2% |
| convnext_tiny320 / dark_0.5 | 89.0% | 3.0% | 90.4% |
| convnext_tiny320 / blur_radius_2 | 88.8% | 3.7% | 92.4% |
| resnet50_320 / clean | 85.9% | 4.7% | 85.3% |
| resnet50_320 / dark_0.5 | 86.1% | 5.4% | 85.6% |
| resnet50_320 / blur_radius_2 | 83.7% | 2.5% | 81.7% |

## Interpretation

The screens are bounded recipes, not exhaustive architecture optimization. The DINO frozen probe has no image augmentation during head fitting. Region crops are fixed to this camera geometry, not detected body parts. The ensemble averages probabilities, and its fold models have no outer-driver training overlap.

Synthetic shift and monitoring checks establish limited robustness and software compatibility, not real-video alert performance. See per-finalist verification.json files for clean/dark/blur, CPU/MPS parity and measured latency. The final five-driver holdout remains unused.
