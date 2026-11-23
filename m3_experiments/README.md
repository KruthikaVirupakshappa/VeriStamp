# M3 experiments: RQ2 edits, image quality, RQ3 and the pipeline

This folder holds the M3 tasks after training (handover guide, tasks 1 to 11). It only **uses** the
trained model: nothing in `../m3_watermark/` is changed. `m3_watermark.py` is imported from there as
it is, and `best.pt` is only read.

## Before you start (Colab or a local machine; for Kaggle see the next section)

1. Download `results_3_final_model.zip` from the Drive link in `../m3_watermark/README.md` and extract
   `best.pt`. Get `hidden_data/test` (the 500 COCO test photos) from Rishitha.
2. Edit the paths at the top of `settings.py` once. Every notebook reads them from there.
3. `pip install diffusers transformers accelerate scikit-image lpips` (already in `../requirements.txt`).
4. Run the notebooks from this folder, in order.

## Running on Kaggle (recommended)

Kaggle's free weekly GPU quota (about 30 hours) is far more than M3 needs (2 to 3 hours), so no lab
booking is required.

**1. Make one private dataset** called `veristamp-m3-inputs` (*Datasets > New dataset*, upload the
zip). A ready one, `veristamp-m3-inputs.zip`, was built with `prepare_kaggle_inputs.py` from the Drive
download and Kruthika's `data/splits` folder. It holds:

```
m3/best.pt                     M3 final model, epoch 97 (from results_3_final_model.zip)
m3/test_manifest.csv           the 500 COCO val2017 test photos (from test_predictions.csv); the
                               notebooks download them from COCO on first run (Internet on)
code/VeriStamp/                this repo, including m3_experiments/
genimage/splits/               test_seen.csv, test_unseen.csv of M1's final split, paths made relative
```

Still to add for notebooks 10 to 12: the GenImage photos (Kruthika's `MyDrive/Datasets/GenImage_native_n500`)
and M1's files (`experiments/m1_final_n500/hp_fft/best.pt` and `vslib_m1_final` in her shared VeriStamp
folder). Rebuild with `prepare_kaggle_inputs.py` (same output folder, add `--genimage-root`,
`--m1-checkpoint`, `--m1-vslib`; only the 1,300 test photos are copied) and upload a new version:

```
python m3_experiments/prepare_kaggle_inputs.py out/veristamp-m3-inputs --zip \
    --m3-model "<the Drive download>.zip" --test-manifest m3_experiments/m3_test_manifest.csv \
    --genimage-splits <data/splits/..._m1_final_n500> --genimage-root <GenImage_native_n500> \
    --m1-checkpoint <best.pt> --m1-vslib <vslib_m1_final>
```

The layout (the dataset name and any extra top folder from the zip do not matter):

**2. Run everything in one session** with `kaggle_run_all.ipynb`: *File > Import notebook*, *Add input*
the dataset, *Settings*: Accelerator **GPU T4 x2** (the P100 has had problems with recent PyTorch)
and Internet **on** (for pip and the Hugging Face models). Set the switches, then *Save Version >
Save & Run All*. It runs in the background, and the outputs stay with that version:
`m3_results.zip` (all tables and figures) and `executed/` (every notebook with its outputs, for the
team's repo). Each Kaggle notebook has its own `/kaggle/working`, which is why the notebooks are run
from one driver rather than as separate Kaggle notebooks.

For a first try, run `kaggle_run_all.ipynb` interactively with `RUN = ['5_setup_check']` to check that
the numbers match notebook 4 before spending GPU time.

## Notebooks

| # | Notebook | Guide task | GPU? | Time |
|---|---|---|---|---|
| 5 | `5_setup_check.ipynb` | 1. Setup: repeats notebook 4's test and checks the numbers match | No | ~5 min CPU |
| 6 | `6_rq2_screenshot.ipynb` | 2. Screenshot simulation, full sequence and each step alone | No | ~10 min CPU |
| 7 | `7_image_quality.ipynb` | 3. SSIM and LPIPS (and PSNR again) | No | ~5 min CPU |
| 8 | `8_rq2_vae_roundtrip.ipynb` | 4. VAE round trip, with a resize-only control and unstamped photos | Recommended | ~10 min T4, 2–3 h CPU |
| 9 | `9_rq2_inpainting.ipynb` | 5. Inpainting at 10%, 25%, 50% masks | **Yes** | ~1 h T4 for 200 photos, ~2 h for 500 |
| 10 | `10_genimage_check.ipynb` | 6. Stamp on GenImage Real and AI photos | No | ~10 min CPU |
| 11 | `11_rq3_detectors.ipynb` | 7 and 9. RQ3: detector scores before and after stamping | Helpful | minutes GPU, ~30 min CPU |
| 12 | `12_pipeline.ipynb` | 7 and 10. Decode, detect and stamp pipeline and its system metrics | No (VAE pass optional) | ~10 min |
| 13 | `13_report_tables.ipynb` | 8 and 11. Collects every result into the guide's tables | No | seconds |

Shared code: `m3_eval.py` (payloads, edits, scoring, tables), `detectors.py` (dummy, M1, M2
placeholder, late fusion), `veristamp_pipeline.py` (the pipeline; the demo can import it).
`../watermarking/m3_hidden.py` wraps M3 for the team's tensor-based `pipeline.py` and tests.

## GPU plan

Only notebooks 8 and 9 really need a GPU, and both only run models (no training). About 1.5 to 3
hours in total on one T4 or better (at least 8 GB of memory; inpainting runs in float16), well inside
Kaggle's weekly quota and its 12 hour session limit. Notebook 9 saves after every photo within a
session.

## Fixed choices (the same in every notebook)

- Each photo's verdict, P(AI) and image tag come from `payload_for(name)` with seed 42, so a photo has
  the same payload across all edits. The stamped PNGs are made once (in `WORK_DIR/stamped`) and
  reused; a JSON next to them records the checkpoint's SHA-256 so stamps from another model are never
  mixed in.
- Brightness/contrast factors, mask positions and inpainting seeds are fixed per photo.
- The VAE uses the latent mean, so the round trip is the same on every run.
- **Inpainting is reported two ways.** The inpainting model returns a whole new image whose every
  pixel has passed through its VAE, which on its own breaks the stamp. Photo editors paste only the
  redrawn region back. The main result is that composited version (the guide's test); the raw model
  output is an extra row.
- Bit accuracy, valid verdict and invalidation rate are always reported separately, plus "wrong but
  valid" (the checksum passed but the message is wrong), which should stay near 0.

## Inputs from the other workstreams (all in place)

| Input | From | Used in |
|---|---|---|
| `hidden_data/test` list (500 COCO photos), M3 `best.pt` | Rishitha (Drive link in `../m3_watermark/README.md`) | all |
| GenImage test photos: `get_genimage_test_photos.ipynb` (Colab) reads M1's 1,300 test photos from the official archives and checks each against M1's SHA-1 | official GenImage archives | 10, 11, 12 |
| M1 `best.pt` (fold 1, SD v1.4 unseen) and `vslib_m1_final` | Kruthika | 11, 12 |
| M2 `best.pt`, `thresholds.json` (fold `stable_diffusion_v_1_4`) and its test predictions | Supriya | 11, 12 |
| Late Fusion `late_thresholds.json` (same fold) and its test predictions | Supriya | 11, 12 |

`prepare_kaggle_inputs.py` builds the Kaggle dataset from these (`--m1-*`, `--m2-*`,
`--fusion-thresholds`, `--reference-predictions`). Notebook 11 checks, before stamping, that every
detector reproduces its owner's score for each photo. Still a placeholder: M1's Uncertain thresholds
(0.35 / 0.65), which affect only M1's verdict counts.
