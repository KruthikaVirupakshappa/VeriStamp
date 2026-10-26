# VeriStamp

**Generalizable AI-Generated Image Detection with Learned Invisible Watermarks**
DATA 255 Deep Learning Vision · San José State University · Fall 2026 · Group 7

VeriStamp detects whether an image is **Real**, **AI-generated** or **Uncertain**, embeds that
verdict as an invisible learned watermark, and studies how detector and watermark affect each other:

| | Question | Main models |
|---|---|---|
| **RQ1** | Do frequency/noise features or semantic features generalise better to unseen generators? | M1 (ResNet-50 on high-pass + FFT), M2 (frozen CLIP + MLP), Fusion |
| **RQ2** | Does the learned watermark survive normal edits better than AI edits? | M3 (HiDDeN-style) |
| **RQ3** | Does adding the watermark change the detector's prediction? | M1, M2, Fusion + M3 |

> The stamp is **not** cryptographically signed. A valid stamp means "our detector said X when the
> image was stamped". It is not proof that an image is authentic.

## Status

| Part | State |
|---|---|
| Shared preprocessing, split builder, `ManifestDataset` | ✅ working + tested |
| Payload format with CRC-8 validity check | ✅ working + tested |
| Metrics (accuracy, AUROC, bit accuracy, PSNR, SSIM, LPIPS, flip rate, …) | ✅ working + tested |
| Pipeline `check → detect → stamp` | ✅ working with stand-in models |
| Stand-in watermark (classical spread-spectrum) | ✅ working; also a non-learned baseline |
| M1, M2, Fusion, baselines | ⬜ TODO (Workstreams A, B) |
| M3 learned watermark, noise layers, VAE/inpainting edits | ⬜ TODO (Workstream C) |
| RQ1/RQ2/RQ3 scripts, Grad-CAM, demo | ⬜ TODO |

## Quick start

```bash
git clone https://github.com/KruthikaVirupakshappa/VeriStamp.git
cd VeriStamp
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

pytest                                 # unit tests (about 10 s on CPU)
python -m tests.integration --n 20     # end-to-end: stamp -> edits -> check
```

On M-series Macs the code uses `mps` automatically; on Colab/Kaggle it uses `cuda`
(`common/seed.py::get_device`).

## Repository map

| Folder / file | What lives there | Main owner |
|---|---|---|
| `configs/` | YAML configs; every script takes `--config`. `base.yaml` holds shared settings | everyone |
| `data/` | `raw/` and `cache/` are git-ignored; `splits/*.csv` manifests **are** committed | A |
| `preprocessing/` | `transforms.py` (the shared `preprocess()`), `build_splits.py`, `datasets.py`, `frequency.py` | A (frequency: B) |
| `models/` | `base.py` interface + stub, `thresholds.py`, M1, M2, Fusion, baselines | B (baselines: A) |
| `watermarking/` | `payload.py`, `base.py` interface + stand-in, `edits.py`, M3, noise layers | C |
| `training/` | training entry points | B, C |
| `evaluation/` | `metrics.py` (shared), RQ1/RQ2/RQ3 scripts, Grad-CAM | B, C, D |
| `pipeline.py` | decode-detect-stamp integration layer | D |
| `demo/` | demo app | D |
| `experiments/` | run folders (ignored) + `runs_log.csv` (committed) | everyone |
| `checkpoints/` | weights (ignored; shared Drive) + `REGISTRY.csv` | everyone |
| `reports/` | generated tables/figures, IEEE report | everyone |
| `docs/` | `interfaces.md`, `decisions.md` | everyone |
| `tests/` | pytest unit tests + `integration.py` | everyone |

## Conventions everyone follows

- **Images travel as float tensors in 0..1**, `[3, H, W]` or `[B, 3, H, W]`. Each model normalises internally.
- **Label 1 = AI, 0 = Real.** Detectors output `P(AI)`.
- **Real and AI images use identical preprocessing.** Only use `preprocessing.transforms.preprocess`.
- **Thresholds come from the validation split**, never from test.
- **"Valid stamp" is defined only by `watermarking.payload.decode`.**
- **Every run gets a row in `experiments/runs_log.csv`.** Report numbers are generated from saved predictions.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the Git workflow, [`docs/interfaces.md`](docs/interfaces.md)
for the contracts between workstreams and [`docs/decisions.md`](docs/decisions.md) for open team decisions.

## Timeline (from the proposal)

| Dates | Milestone |
|---|---|
| Sep 30 – Oct 7 | Proposal; data preparation, bias control, baselines |
| Oct 7 – Oct 28 | Detector training (RQ1); watermark training (RQ2) |
| Oct 28 | Interim: Fusion vs ResNet-50 on held-out generators; watermark >95% bits after JPEG |
| Oct 28 – Nov 18 | RQ2 and RQ3 experiments; integration and demo |
| Nov 24 / Dec 2 | Code, report and slides due / final presentation |
