# Fusion

Fusion combines the outputs of M1 and M2 to detect whether an image is real or AI-generated.

M1 is the high-pass and FFT ResNet detector. It produces a 2,048-dimensional feature vector. M2 is the frozen CLIP detector. It produces a 512-dimensional image embedding. Fusion uses both representations for the same image, after checking that image IDs, labels, and row order match.

## Dataset and evaluation setup

Fusion uses the same authoritative M1 splits and preprocessing as the other project components. Each fold has:

- 1,400 training images
- 300 validation images
- 300 seen-generator test images
- 1,000 unseen-generator test images

The three leave-one-generator-out folds are:

1. ADM and VQDM are used for training, with Stable Diffusion v1.4 held out.
2. Stable Diffusion v1.4 and VQDM are used for training, with ADM held out.
3. Stable Diffusion v1.4 and ADM are used for training, with VQDM held out.

The validation split is used for model selection and for choosing the Real, AI, and Uncertain thresholds. The unseen test split is used only for final evaluation.

## Early Fusion

Early Fusion standardizes the M1 and M2 features using training data only. It then concatenates them into one 2,560-dimensional vector:

```text
2,048 M1 features + 512 M2 features = 2,560 fused features
```

An MLP classifier is trained on this fused vector. M1 and M2 are kept frozen during Fusion training. The baseline configuration uses a 512-unit hidden layer, 0.3 dropout, and weight decay of `1e-4`.

## Late Fusion

Late Fusion is included as a comparison method. It does not train another feature-level classifier. Instead, it averages the M1 and M2 AI probabilities:

```text
late probability = (M1 probability + M2 probability) / 2
```

This tests whether combining the final decisions is more reliable than combining the internal feature vectors.

## Normal-image results

The table shows AUROC on the unseen generator in each fold.

| Held-out generator | Early Fusion | Late Fusion |
|---|---:|---:|
| Stable Diffusion v1.4 | 0.7393 | 0.8780 |
| ADM | 0.8775 | 0.9445 |
| VQDM | 0.9044 | 0.9533 |
| Mean | 0.8404 | 0.9252 |
| Sample standard deviation | 0.0886 | 0.0412 |

Early Fusion performs better on the seen-generator test sets than on the unseen generators, but its training AUROC is almost or exactly 1.0. This large training-to-unseen gap indicates overfitting and suggests that the MLP is learning feature combinations that do not transfer equally well to a new generator.

Late Fusion gives the strongest and most consistent unseen-generator results. Its mean AUROC is 0.9252, compared with 0.8404 for Early Fusion. Its smaller standard deviation also shows that it varies less between generator folds.

## Three-way decisions

The system supports three decisions:

- Real
- AI
- Uncertain

The lower and upper thresholds are fitted separately for Early Fusion and Late Fusion using validation predictions only. Scores below the lower threshold are classified as Real, scores above the upper threshold are classified as AI, and scores between the thresholds are classified as Uncertain.

The test sets are not used to select thresholds. The threshold files and three-way metrics are saved separately for every fold.

## Exact JPEG robustness evaluation

JPEG degradation was evaluated at quality 65 and quality 30. For each original image, the complete original image was re-encoded as JPEG using 4:2:0 subsampling. The decoded image was then passed through the same native centre-crop preprocessing used by the detectors.

For Early Fusion, this is a feature-level evaluation. New degraded M1 features and new degraded M2 embeddings were extracted before applying the trained Fusion classifier. For Late Fusion, new M1 and M2 probabilities were computed from those same degraded images and then averaged. The existing clean-image probability CSVs were not reused.

### Early Fusion JPEG results

| Quality | Split | Stable Diffusion held out | ADM held out | VQDM held out | Mean | Sample standard deviation |
|---|---|---:|---:|---:|---:|---:|
| JPEG65 | test_seen | 0.9362 | 0.9187 | 0.9048 | 0.9199 | 0.0157 |
| JPEG65 | test_unseen | 0.6964 | 0.8498 | 0.8963 | 0.8142 | 0.1046 |
| JPEG30 | test_seen | 0.8784 | 0.8741 | 0.8348 | 0.8624 | 0.0241 |
| JPEG30 | test_unseen | 0.6179 | 0.7647 | 0.8461 | 0.7429 | 0.1156 |

### Late Fusion JPEG results

| Quality | Split | Stable Diffusion held out | ADM held out | VQDM held out | Mean | Sample standard deviation |
|---|---|---:|---:|---:|---:|---:|
| JPEG65 | test_seen | 0.9637 | 0.9491 | 0.9377 | 0.9501 | 0.0130 |
| JPEG65 | test_unseen | 0.8073 | 0.9005 | 0.9362 | 0.8813 | 0.0665 |
| JPEG30 | test_seen | 0.9213 | 0.9139 | 0.8772 | 0.9042 | 0.0236 |
| JPEG30 | test_unseen | 0.6682 | 0.8018 | 0.8579 | 0.7760 | 0.0974 |

JPEG65 causes a moderate drop on the unseen tests. JPEG30 causes a larger drop, especially for the Stable Diffusion fold. Late Fusion remains stronger than Early Fusion under both JPEG qualities, although it also loses performance as compression becomes stronger. This means that compression removes or changes some of the information used by the detectors.

## Files and outputs

The baseline Fusion checkpoints and normal-image results are saved under:

```text
experiments/fusion_m1_m2_aligned_n500_seed42/
```

Each fold contains:

- `best.pt`, the trained Early-Fusion checkpoint
- `metrics.json`, including Early Fusion, Late Fusion, and three-way metrics
- `early_thresholds.json` and `late_thresholds.json`
- prediction CSV files for train, validation, seen-test, and unseen-test

The exact JPEG evaluation outputs are saved under:

```text
experiments/fusion_jpeg_m1_m2_aligned_n500_seed42/
```

The JPEG output contains the degraded M1 and M2 feature arrays, row-alignment indexes, Early-Fusion prediction files, Late-Fusion prediction files, and metrics for both test splits and both JPEG qualities.

The main training script is:

```text
training/train_fusion.py
```

The exact JPEG evaluation script is:

```text
training/evaluate_fusion_jpeg.py
```

## Conclusion

Fusion was completed for all three leave-one-generator-out folds, including validation-based three-way decisions and exact JPEG65/JPEG30 evaluation for both Early and Late Fusion.

Early Fusion demonstrates that feature concatenation is possible, but its nearly perfect training performance and lower unseen-generator AUROC show that it overfits in this dataset size and setup. Late Fusion is the more reliable Fusion method because it has higher mean unseen AUROC and lower variation across folds.

Therefore, the project should report Early Fusion as the feature-level Fusion experiment and Late Fusion as the stronger comparison or practical Fusion result. M2 alone still remains an important baseline because it performed strongly on unseen generators without requiring the additional Fusion head.
