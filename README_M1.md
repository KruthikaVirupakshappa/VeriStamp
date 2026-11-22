# M1: frequency/noise ResNet-50 detector

**VeriStamp · DATA 255 · Group 7**

M1 decides whether an image is **Real (0)** or **AI-generated (1)** and outputs **P(AI)**. It is an ImageNet-pretrained **ResNet-50** that does not look at the ordinary colour image. Its input is:
- a **high-pass residual**: the fine detail left after blurring is subtracted;
- an **FFT log-magnitude spectrum**: which spatial frequencies the image contains.

The question M1 answers for VeriStamp (RQ1): *do frequency/noise features detect images from a generator the model never saw?* An RGB ResNet-50, trained identically, is the baseline.

- **Full analysis:** experiments, results, shortcut analysis and conclusions are in [`docs/M1_ANALYSIS.md`](docs/M1_ANALYSIS.md).
- **This file:** the final result, how to reproduce it, and what M2/Fusion should reuse.

---

## Final result: three leave-one-generator-out folds

The final setup was run three times, holding out a different generator each time. Everything else is identical:
- the same 7,800 images, 500 per class per generator;
- seed 42;
- native 224 crops with equal JPEG;
- the same model and training settings.

| Fold | Run | Seen (train/val/test_seen) | Unseen (test only) | RGB AUROC | M1 AUROC | M1 − RGB [95% CI] |
|---|---|---|---|---|---|---|
| 1 | `m1_final_n500` | ADM, VQDM | **Stable Diffusion v1.4** | 0.681 | 0.712 | **+0.031 [0.001, 0.059]** |
| 2 | `m1_final_n500_unseen_adm` | VQDM, SD v1.4 | **ADM** | 0.816 | 0.866 | **+0.050 [0.028, 0.072]** |
| 3 | `m1_final_n500_unseen_vqdm` | ADM, SD v1.4 | **VQDM** | 0.886 | 0.889 | +0.004 [−0.016, 0.023] |
| **Mean** | | | | **0.794** | **0.822** | **+0.028** |

(n = 1,000 unseen images per fold. Paired bootstrap on the same images.)

**In plain words:**
- On the unseen generator, M1 was at least as good as RGB in every fold. It was clearly better on Stable Diffusion and ADM, and equal on VQDM.
- The folds are not equally hard. VQDM is the easiest held-out generator (both models score about 0.89), probably because it resembles ADM. So report the per-fold rows first and the mean second.
- On **seen** generators there is no M1 advantage in any fold. The test_seen differences are −0.010, −0.009 and −0.009, all with intervals that include 0.
- **Strong JPEG compression (quality 30) removes M1's advantage in every fold:** +0.007, 0.000 and +0.012, all with intervals that include 0.

One seed and 500 per class: interim evidence, not a final benchmark. See [Section 10 of the analysis](docs/M1_ANALYSIS.md#10-final-m1-conclusion).

### Fold 1 in detail (`m1_final_n500`)

Setup:
- GenImage subset, 500 images per class per generator (interim level).
- Trained on ADM + VQDM; Stable Diffusion v1.4 held out as the unseen test generator.
- Seed 42.
- Native 224×224 crops (no resizing) and equal JPEG for Real and AI.

| Test set | n | RGB AUROC | M1 AUROC | M1 − RGB [95% CI] | M1 accuracy | M1 recall (AI caught) |
|---|---|---|---|---|---|---|
| Seen generators (ADM, VQDM) | 300 | 0.947 | 0.937 | −0.010 [−0.033, 0.013] | 0.890 | 0.907 |
| **Unseen Stable Diffusion v1.4** | 1000 | 0.681 | **0.712** | **+0.031 [0.001, 0.059]** | 0.622 | 0.338 |
| Unseen SD after JPEG quality 65 | 1000 | 0.642 | 0.671 | +0.029 [−0.001, 0.058] | 0.570 | 0.220 |
| Unseen SD after JPEG quality 30 | 1000 | 0.578 | 0.585 | +0.007 [−0.025, 0.038] | 0.529 | 0.126 |

**In plain words:**
- M1 is as good as RGB on familiar generators.
- On the unseen generator, M1 is slightly better than RGB. The advantage is small, and its lower bound is near 0.
- Both models still miss most Stable Diffusion images at the default 0.5 threshold.
- Strong JPEG compression removes M1's advantage.

Folds 2 and 3 have the same tables (seen, unseen, JPEG 65/30, Grad-CAM) in their own `M1_FINAL_SUMMARY.md`. A summary is in [Section 7.7 of the analysis](docs/M1_ANALYSIS.md#77-three-fold-evaluation).

---

## End-to-end pipeline

```mermaid
flowchart TD
    A["GenImage official archives on Drive<br/>Datasets/GenImage_official/&lt;generator&gt;/*.zip"] -->|"Section 0: extract 650 per folder"| B["Datasets/GenImage_native_n500<br/>ADM, VQDM, SD v1.4 · ai/ and nature/"]
    B -->|"label check (sample grid, LABELS_CONFIRMED)"| C["Manifest<br/>image_id, path, label, generator, SHA-1, size, format"]
    C -->|"duplicates (SHA-1, dHash), drop sides < 224 px"| D["Fixed split, seed 42<br/>train 1400 · val 300 · test_seen 300 · test_unseen 1000"]
    D -->|"leakage checks: all PASS"| E["Frozen preprocessing<br/>native size → JPEG 75–95 (4:2:0) → central tile → 224 crop"]
    E --> F1["RGB baseline<br/>ResNet-50, RGB input"]
    E --> F2["M1<br/>ResNet-50, high-pass + FFT input"]
    F1 --> G["best.pt (highest val AUROC)"]
    F2 --> G
    G --> H["Predictions: val, test_seen, test_unseen<br/>+ JPEG 65 / 30 versions of the test sets"]
    H --> I["Metrics, paired bootstrap comparisons,<br/>plots, Grad-CAM, M1_FINAL_SUMMARY.md"]
```

---

## Reproducibility

### What you need
- A Google account with the GenImage archives in Drive at `MyDrive/Datasets/GenImage_official/`, in folders named `ADM/`, `VQDM/` and `stable_diffusion_v_1_4/`, containing the official multi-part zip files (for example `imagenet_ai_0508_adm.z01 … .zip`).
  - Get them from the official source <https://github.com/GenImage-Dataset/GenImage> (Google Drive link, or Baidu mirror with share code `ztf1`). No passwords or API keys are needed.
  - Only the selected images are extracted; the archives are never fully unpacked.
- The `m1_run_001` reference split at `MyDrive/VeriStamp/data/splits/genimage_unseen-stable_diffusion_v_1_4_n200_seed42/`. Keep it there for an exact reproduction: the final split reuses its Stable Diffusion test images.
- **Google Colab with an A100 GPU** (all runs used an NVIDIA A100-SXM4-80GB, bfloat16 mixed precision).
  - Training takes under a minute per model at 500 per class.
  - Extracting the images (Section 0) and the JPEG-degradation caches take longer.
  - The notebook also runs on CPU (batch size 8), which is enough for the sanity checks but not for real training.

### Steps
1. Open `notebooks/M1_final_n500.ipynb` in Colab. Choose **Runtime → Change runtime type → A100 GPU**.
2. Run **Section 0**:
   - it mounts Drive, checks which generators exist (VQDM replaces BigGAN), and extracts 650 images per image folder into `MyDrive/Datasets/GenImage_native_n500/`;
   - it stops at **0.11**. Look at the sample grid in **0.9** (AI images should look generated, Real ones like photos), set `LABELS_CONFIRMED = True`, and rerun 0.11 and 0.12.
3. Run **Sections 1–27** in order. Nothing else needs editing.
4. **Folds 2 and 3:** open `notebooks/M1_folds_n500.ipynb`, set `FOLD = 2` (unseen ADM) or `FOLD = 3` (unseen VQDM), and run all.
   - There is no Section 0: this notebook reuses the fold 1 dataset folder and only reads it.
   - Fold 1 must exist first: its split is read to confirm the settings are identical.
5. Check these outputs:

| Section | Expected |
|---|---|
| 7 | Every leakage check PASS. Split sizes 1400 / 300 / 300 / 1000 |
| 7b | Fold 1: "Every usable m1_run_001 unseen image is in test_unseen" PASS (391 of 400). Folds 2/3: "Unseen generator is the FOLD choice" and "Same settings as fold 1" PASS |
| 14 | ALL SANITY CHECKS PASSED |
| 27 | ALL CHECKS PASSED |

If a Colab session fails with `GET was unable to find an engine` at the first convolution, the runtime's cuDNN library is broken. Disconnect and delete the runtime, reconnect, and rerun from Section 1.

### Configuration (final run)

| Setting | Value |
|---|---|
| `EXPERIMENT` / run name | `m1_final_n500` |
| `DATA_ROOT` | `/content/drive/MyDrive/Datasets/GenImage_native_n500` |
| `PROJECT_DIR` | `/content/drive/MyDrive/VeriStamp` |
| `N_PER_CLASS` | 500 (subset level `interim`) |
| Seed | 42 |
| Generators | Seen ADM, VQDM; unseen `stable_diffusion_v_1_4` |
| Split fractions (seen) | train 0.70, val 0.15, test_seen 0.15; oversample 1.3; near-duplicates dropped |
| Preprocessing | `mode: native`, `crop_size: 224`, `tile_size: 256`, `align: 8`, `jpeg_quality_range: [75, 95]`, `jpeg_subsampling: 4:2:0`, `min_side: 224`, `train_hflip: True`, no random rescale |
| Features | high-pass Gaussian kernel 5, σ 1.0; scale statistics from 20 training batches |
| Model | ResNet-50, `IMAGENET1K_V2` weights, dropout 0.3; modes `rgb` and `hp_fft` |
| Training | ≤ 10 epochs, batch 64, AdamW (lr 1e-4 backbone, 1e-3 new layers, weight decay 1e-4), 1 warm-up epoch + cosine, early stop patience 3 |
| Sanity | overfit 32 images (80 steps), shuffled labels (512 images, 2 epochs) |
| JPEG degradation | qualities 65 and 30, 4:2:0, on `test_seen` and `test_unseen` |
| Grad-CAM | layer `backbone.layer4`, 4 examples per group, 64 images per average |

Every run saves its full configuration, plus the environment and split fingerprints, as `config.yaml` in each model folder.

### Run names

| Run | Executed notebook | In git? | Role |
|---|---|---|---|
| `m1_run_001` | `M1_run001_executed.ipynb` | No (kept locally / Drive) | Original baseline (resize; ADM + BigGAN) |
| `m1_random_rescale` | `M1_random_rescale.ipynb` | No | Random-rescale attempt (M1 only) |
| `m1_resize_vqdm` | `M1_resize_vqdm_executed.ipynb` | No | Control: VQDM replaces BigGAN, old resize |
| `m1_native_crop_equal_jpeg` | `M1_native_crop_equal_jpeg_executed.ipynb` | No | Native crop + equal JPEG, 200 per class |
| **`m1_final_n500`** | `M1_final_n500_executed.ipynb` | **Yes, as evidence** | **Final M1, fold 1** (unseen SD v1.4), 500 per class |
| **`m1_final_n500_unseen_adm`** | `M1_fold2_unseen_adm_executed.ipynb` | **Yes, as evidence** | **Final M1, fold 2** (unseen ADM) |
| **`m1_final_n500_unseen_vqdm`** | `M1_fold3_unseen_vqdm_executed.ipynb` | **Yes, as evidence** | **Final M1, fold 3** (unseen VQDM) |

Two final-run notebooks are in git:
- **`notebooks/M1_final_n500_executed.ipynb`** is the evidence: the actual run with all outputs (metrics, plots, Grad-CAM, final verification). Read it; don't rerun it.
- **`notebooks/M1_final_n500.ipynb`** is the one to run. It has the same code with outputs cleared and `LABELS_CONFIRMED = False`, so the label check is not skipped.

For folds 2 and 3, `notebooks/M1_folds_n500.ipynb` is the one to run (outputs cleared). The two `M1_fold*_executed.ipynb` files are the evidence.

The four earlier runs are summarised in [`docs/M1_ANALYSIS.md`](docs/M1_ANALYSIS.md), and their result files are on Drive.
- Each run writes to its own folders. Earlier runs are recorded in a snapshot and checked as unchanged at the end.
- Never reuse a run name for new settings. Choose a new `EXPERIMENT` name instead.

### Resuming an interrupted run
- `last.pt` is saved after every epoch.
- To continue the **same** experiment after a disconnect: set `RESUME = True` in Section 1 (same `EXPERIMENT`), then rerun from Section 1.
  - The split is reloaded (fingerprints verified).
  - The caches are copied back from Drive.
  - Training continues from the next epoch.
- Leave `RESUME = False` for a fresh run. If the run folder already exists, the notebook stops with `FileExistsError` instead of overwriting it.

### Where everything is saved (Google Drive)

```
MyDrive/VeriStamp/
├── data/splits/genimage_seen-ADM-VQDM_unseen-stable_diffusion_v_1_4_min224_n500_seed42__m1_final_n500/
│   ├── manifest.csv  split_info.json  train.csv  val.csv  test_seen.csv  test_unseen.csv
│   └── leakage_checks.csv  reference_unseen_missing.csv
├── data/manifests/m1_final_n500/          dropped_too_small.csv, cross_archive_duplicates.csv, ...
├── cache/<split name>/                     {train,val,test_seen,test_unseen}_native256.npy (preprocessed crops)
├── vslib_m1_final/                         the Python modules the notebook wrote (vs_*.py)
├── vslib_m1_folds/                         the same modules, written by the fold 2/3 notebook
├── features/m1_final_n500{,_unseen_adm,_unseen_vqdm}_hp_fft/   M1 2048-d features, one folder per fold
└── experiments/
    ├── experiments_log.csv                 one row per trained model, all runs
    └── m1_final_n500/
        ├── README.md  results_summary.csv/.md  generalization_gap.csv
        ├── M1_FINAL_SUMMARY.md  m1_final_summary.json  final_verification.json  protected_snapshot.json
        ├── plots/roc_comparison.png  data_plots/
        ├── data_checks/                    jpeg_audit.csv, jpeg_checks.csv, crop_coverage.csv, original_info_<split>.csv
        ├── comparison/                     comparison_results.csv/.md, comparison_pairs.csv, comparison_plots.png
        ├── jpeg_degradation/               degradation_results.csv, degradation_pairs.csv, degradation_auroc.png
        ├── gradcam/                        gradcam_<group>.png ×5, gradcam_average.png/.csv, gradcam_images.csv
        ├── rgb/                            (same files as hp_fft/)
        └── hp_fft/                         M1
            ├── config.yaml  best.pt (94 MB)  last.pt  history.csv  metrics.json  README.md
            ├── predictions_val.csv  predictions_test_seen.csv  predictions_test_unseen.csv
            ├── predictions_test_seen_jpeg65.csv  predictions_test_seen_jpeg30.csv
            ├── predictions_test_unseen_jpeg65.csv  predictions_test_unseen_jpeg30.csv
            ├── unseen_test_log.json  shortcut_diagnostic.json (17b)
            ├── diagnostics/                17c: by_original_format.csv, real_by_original_jpeg_quality.csv, ...
            └── plots/                      history.png, roc.png, pr.png, confusion_*.png, scores_*.png,
                                            examples_correct_unseen.png, examples_incorrect_unseen.png
```

Folds 2 and 3 use the same layout:
- run folders `experiments/m1_final_n500_unseen_adm/` and `experiments/m1_final_n500_unseen_vqdm/`;
- splits `data/splits/genimage_seen-VQDM-stable_diffusion_v_1_4_unseen-ADM_min224_n500_seed42__m1_final_n500_unseen_adm/` and `data/splits/genimage_seen-ADM-stable_diffusion_v_1_4_unseen-VQDM_min224_n500_seed42__m1_final_n500_unseen_vqdm/`;
- caches under `cache/<split name>/`;
- each fold's `comparison/cross_fold_summary.csv` holds the cross-fold table.

Not in git:
- Checkpoints and data are git-ignored and live only on the team Drive.
- The repository's `models/m1_resnet.py` is still a stub. The working M1 code is the `vs_*.py` library the notebook writes (copied to `VeriStamp/vslib_m1_final/`). Moving it into the repository is still to do.

### How a teammate reproduces it
1. Ask for access to the team Drive folders `Datasets/GenImage_official/` and `VeriStamp/data/splits/genimage_unseen-stable_diffusion_v_1_4_n200_seed42/`, or download the three generator archives from the official source into the same layout.
2. Run `notebooks/M1_final_n500.ipynb` exactly as in [Steps](#steps), in **your own Drive**.

   Do not point `PROJECT_DIR` at someone else's finished `m1_final_n500` folder: the notebook will refuse to overwrite it.
3. Compare your Section 19 and Section 23 tables with the final-result table above.

   The data sampling is seeded (42), so you should get the same split. Small differences in the last decimals of the metrics are possible because GPU training is not bit-for-bit deterministic. This was not measured.

---

## Handover for M2 and Fusion

The aim is that M2 and Fusion are compared with M1 on **exactly the same images, labels and preprocessing**, **in each of the three folds**.
- Every item below exists once per fold: replace `m1_final_n500` with `m1_final_n500_unseen_adm` or `m1_final_n500_unseen_vqdm`, and use that fold's split folder.
- Never mix folds. For example, fold 2 Fusion uses only fold 2 M1 features and fold 2 M2 outputs.

| Item | Where | How to reuse it |
|---|---|---|
| **Manifest** | `data/splits/<final split>/manifest.csv` | One row per image: `image_id`, `image_path`, `label`, `generator`, `file_hash` (SHA-1), `width`, `height`, `file_format`, `split`, plus status columns. Do not rebuild it |
| **Fixed split** | `train.csv`, `val.csv`, `test_seen.csv`, `test_unseen.csv`, `split_info.json` | Use these files as they are. `split_info.json` holds fingerprints; check them before training. The fold's unseen generator (SD v1.4, ADM or VQDM) must never appear in training or validation |
| **Labels** | `label` column | **Real = 0, AI = 1**. Detectors output P(AI) |
| **Preprocessing** | `vs_native.py` in `vslib_m1_final/`; caches in `cache/<final split>/*_native256.npy` | Same frozen pipeline: native size, JPEG 75–95 (4:2:0, seeded by image ID), central 256 tile aligned to 8. Training: random 224 crop + flip; evaluation: centre 224 crop. Images are 0..1 float tensors; each model normalises internally (CLIP needs its own normalisation and input size inside M2) |
| **M1 checkpoint** | `experiments/m1_final_n500/hp_fft/best.pt` (RGB: `rgb/best.pt`) | A dict with `model` (state dict), `epoch`, `val_metrics`, …. Load with `vs_models.build_detector("hp_fft", CONFIG)` + `load_state_dict(ck["model"])`, as the notebook's `load_best()` does. The reload reproduces the validation AUROC exactly (0.9518) |
| **M1 predictions** | `hp_fft/predictions_{val,test_seen,test_unseen}.csv` (+ `_jpeg65`, `_jpeg30` for the test sets) | Columns: `split`, `image_id`, `image_path`, `generator`, `label`, `logit`, `p_ai`, `pred`, `model`, `correct`, `cache_index`. Join with M2 on `image_id` |
| **2048-d feature vectors** | `features/m1_final_n500_hp_fft/` (made by `notebooks/M1_extract_features.ipynb`) | `<split>_features.npy`: float32 [N, 2048], row i = row i of the split CSV. `<split>_index.csv` gives `cache_index`, `image_id`, `generator`, `label`, `logit`, `p_ai` for the same rows. Details below |
| **Evaluation code** | `vs_metrics.py`; Sections 19, 23 and 24 of the final notebook | `classification_metrics(labels, p_ai)` gives accuracy, AUROC, precision, recall, F1 and the confusion matrix at 0.5. Section 23 has the paired bootstrap AUROC comparison (2,000 resamples, stratified). Section 24 has the JPEG 65/30 test |
| **Metrics and plots** | `metrics.json`, `results_summary.csv`, `comparison/`, `jpeg_degradation/`, `gradcam/`, `plots/` | M1 reference numbers for the RQ1 table |

**Rules to keep the comparison fair:**
- Any decision threshold (for example for Real / Uncertain / AI) must be chosen on **validation** data only.
- `test_unseen` must not be used for tuning, in any fold. The Stable Diffusion test set has already been used once for the final preprocessing choice; future decisions should be based on validation data.
- Report M2 and Fusion with the same paired bootstrap against M1 on the same images.

### The saved 2048-d features
`notebooks/M1_extract_features.ipynb` loaded each fold's `hp_fft/best.pt` (no training) and saved one 2,048-number vector per image. The vector is the pooled ResNet-50 output, i.e. the input of M1's final layer. Shapes in every fold: train (1400, 2048), val (300, 2048), test_seen (300, 2048), test_unseen (1000, 2048).

AUROC from the saved features vs AUROC in the run's saved predictions:

| Fold | Folder under `features/` | train | val | test_seen | test_unseen |
|---|---|---|---|---|---|
| 1 | `m1_final_n500_hp_fft` | 0.9999 | 0.9511 vs 0.9518 | 0.9374 vs 0.9371 | 0.7123 vs 0.7117 |
| 2 | `m1_final_n500_unseen_adm_hp_fft` | 0.9999 | 0.9290 vs 0.9296 | 0.9191 vs 0.9188 | 0.8655 vs 0.8656 |
| 3 | `m1_final_n500_unseen_vqdm_hp_fft` | 1.0000 | 0.9180 vs 0.9174 | 0.9350 vs 0.9341 | 0.8899 vs 0.8895 |

(Train predictions were not saved during training, so there is nothing to compare the train features against.)

- **All splits use the evaluation crop:** the centre 224 crop of the valid region, offset a multiple of 8. No augmentation, train included. Computed in float32; the training run scored in bfloat16, hence the small AUROC differences.
- **Checks passed:**
  - cache fingerprints match the split CSVs;
  - logit correlation with the saved predictions ≥ 0.9995 in every fold, and Real/AI prediction agreement ≥ 98.3%;
  - same image order, no NaN, and the reloaded files match what was computed.
- **Provenance:** `features_info.json` records the checkpoint SHA-256 (fold 1 `5b4d9baa…167cd`, fold 2 `cc5d7072…8615e`, fold 3 `f92389d9…317a4`), split fingerprints and preprocessing. `feature_checks.csv` holds the comparison table.
- **Known mislabel:** the `README.md` written inside the fold 2 and 3 feature folders says "test_unseen (Stable Diffusion v1.4)". In those folds `test_unseen` is ADM and VQDM respectively. `features_info.json` (`split_name`) is correct.
- **Library folder:** the fold 2/3 extractions loaded the code from `vslib_m1_final`. It's the same code as `vslib_m1_folds`, and the scores reproduced the saved predictions.
- **Caution for Fusion:**
  - M1 almost perfectly separates its own training images (train AUROC 0.9999–1.0000 in every fold), so train-split features are far more confident than on new images.
  - A Fusion model trained on them may learn to over-trust M1.
  - Consider fitting Fusion on validation features (with cross-validation), or at least checking for this on val.

---

## Evidence

Every number in this file and in `docs/M1_ANALYSIS.md` comes from the printed outputs of:
- `notebooks/M1_run001_executed.ipynb`
- `notebooks/M1_random_rescale.ipynb`
- `notebooks/M1_extract_features.ipynb`, `M1_extract_features_unseen_adm.ipynb` and `M1_extract_features_unseen_vqdm.ipynb` (2048-d features, folds 1–3)
- `notebooks/M1_fold2_unseen_adm_executed.ipynb` and `M1_fold3_unseen_vqdm_executed.ipynb` (folds 2 and 3: Sections 7, 14, 19, 22, 23/23b, 24, 25, 27)
- `notebooks/M1_resize_vqdm_executed.ipynb`
- `notebooks/M1_native_crop_equal_jpeg_executed.ipynb`
- `notebooks/M1_final_n500_executed.ipynb` (committed to git, like the fold 2/3 and feature notebooks; the four earlier runs are not), especially:
  - Section 19 (metrics);
  - Section 23 (paired comparisons across all runs);
  - Section 24 (JPEG degradation);
  - Section 25 (Grad-CAM);
  - Section 26 (`M1_FINAL_SUMMARY.md`);
  - Section 27 (final verification: ALL CHECKS PASSED).

These outputs are generated from the saved Drive files (`metrics.json`, `predictions_*.csv`, `comparison/*.csv`, `jpeg_degradation/*.csv`, `gradcam/gradcam_average.csv`). The Drive files themselves were not opened separately when writing this document.
