# M2: Frozen CLIP Detector

M2 uses a frozen CLIP ViT-B/16 image encoder followed by a small trainable MLP classifier. The same M1 CSV splits and labels were used so that the results could be compared fairly with M1.

## Evaluation setup

The final evaluation uses three leave-one-generator-out folds:

1. Stable Diffusion v1.4 held out
2. ADM held out
3. VQDM held out

Each fold contains 1,400 training images, 300 validation images, 300 seen-test images, and 1,000 unseen-test images. Native preprocessing and equal JPEG preprocessing were kept consistent across the folds. The caches were checked so that their row order matched the split CSV files, and the held-out generator was not included in training or validation.

## What we did for M2

The M2 pipeline was completed using the following steps:

1. Reused the final M1 CSV splits instead of creating new splits.
2. Rebuilt native 256 pixel caches from the raw images in the same row order as the M1 CSV files.
3. Checked image IDs, labels, cache shapes, missing files, and split overlap before training.
4. Used a frozen CLIP ViT-B/16 image encoder to extract 512-dimensional image embeddings.
5. Trained a small MLP classifier on the training embeddings.
6. Selected the best checkpoint using validation AUROC and early stopping.
7. Evaluated the model on the seen and unseen test sets for each fold.
8. Evaluated JPEG quality 65 and 30 by compressing the original image first and then applying the normal native preprocessing.
9. Added validation-based thresholds for Real, AI, and Uncertain predictions.

The main training command is:

```bash
.venv/bin/python -m training.train_detector \
  --config configs/m2.yaml \
  --fold-spec data/splits/m2_m1_aligned_n500_seed42/folds.json \
  --fold all \
  --run-dir experiments/m2_m1_aligned_n500_seed42 \
  --workers 0
```

The fold-specific checkpoints, embeddings, predictions, thresholds, and metrics are saved inside the run directory. The generated cache and experiment files are kept outside version control because they are large and can be regenerated from the split files and raw dataset.

## Baseline results

| Held-out generator | Unseen AUROC |
|---|---:|
| Stable Diffusion v1.4 | 0.9064 |
| ADM | 0.9524 |
| VQDM | 0.9451 |
| Mean | 0.9364 |

The results show that M2 generalizes well to generators that were not used during training. Stable Diffusion was the most difficult held-out generator, while ADM and VQDM gave stronger results.

## JPEG robustness and three-way decisions

M2 was also evaluated after applying JPEG quality 65 and 30 to the original images before the normal native preprocessing step. The model uses validation data to select thresholds for three possible decisions: Real, AI, and Uncertain. The test sets were not used to select these thresholds.

Performance decreased at JPEG quality 30, especially for the Stable Diffusion fold. This shows that compression still affects the detector and should be considered when interpreting predictions.

| Held-out generator | JPEG65 AUROC | JPEG30 AUROC |
|---|---:|---:|
| Stable Diffusion v1.4 | 0.8224 | 0.6743 |
| ADM | 0.8859 | 0.7918 |
| VQDM | 0.9059 | 0.8020 |

## Regularization experiments

Two training variations were tested to reduce the train-validation gap. Stronger image augmentation reduced unseen-generator performance, especially for Stable Diffusion. A lighter augmentation setup also performed below the baseline.

A regularization-only setup with higher dropout and weight decay produced almost the same mean AUROC as the baseline. Therefore, the original baseline configuration was kept as the final M2 model, while the other settings were treated as ablation experiments.

Overall, the results suggest that M2 learns useful features that transfer across generators, although generator differences and JPEG compression can still affect performance.
