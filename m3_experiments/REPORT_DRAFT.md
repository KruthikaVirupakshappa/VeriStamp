# Report sections: RQ2 edits, image quality, RQ3 and the pipeline

Text for the report sections the handover guide assigns to M3 (tasks 8 and 11), filled in from the
Kaggle run of 20 November 2026 (Tesla T4; `results/`, notebooks in `executed/`). Every number below is
in `results/report_tables.md` or the per-photo CSVs next to it. Two places still depend on the team:
M2 and the fusion model (RQ3 was run on M1 only), and the Uncertain thresholds (placeholders 0.35 and
0.65; they affect only the verdict counts in RQ3 and the pipeline, not the score changes).

---

## RQ2: does the watermark survive normal edits better than AI edits?

### Setup

All RQ2 tests use the 500 COCO val2017 test photos, which were never used for training or for choosing
the model, at their original size. Each photo is stamped once with a fixed 32-bit message (a verdict,
a P(AI) bucket, a detector ID, a 14-bit image tag and a CRC-8 checksum), chosen per photo with a fixed
seed so that every edit is applied to the same stamped photo. After each edit the message is read back
and we report bit accuracy (the share of the 32 bits read correctly; 50% is guessing), valid verdict
(the share of photos whose message passes the checksum and gives the right verdict) and the stamp
invalidation rate (the share whose checksum fails). Bit accuracy and valid verdict are reported
separately because a few wrong bits are enough to make the checksum reject a message.

**Normal edits.** JPEG at quality 90, 70 and 50 and resizing to 224 pixels, as in the model's own test.
The screenshot simulation resizes the photo to 80%, applies a Gaussian blur of radius 0.5, changes
brightness and contrast by a fixed random factor of up to 5% each, resizes back and saves as PNG; each
step was also tested on its own.

**AI edits.** The VAE round trip passes the photo through the image encoder and decoder of Stable
Diffusion (`stabilityai/sd-vae-ft-mse`) at 512 by 512 pixels, which redraws every pixel; a control with
the same resizing but no VAE separates the two effects. For inpainting, Stable Diffusion 1.5 inpainting
(`stable-diffusion-v1-5/stable-diffusion-inpainting`; the version 2 model planned in the proposal has
been withdrawn from Hugging Face) redraws one rectangle covering 10%, 25% or 50% of the photo at a fixed
random position, with the prompt "a natural photo", 30 steps and a fixed seed. The redrawn region is
pasted back into the stamped photo, so pixels outside the mask are unchanged, as in a photo editor.
Inpainting used a fixed subset of 200 of the 500 photos, chosen by a seeded shuffle (the handover
guide's minimum, to keep GPU time down).

### Results

| Edit | Bit accuracy | Valid verdict | Stamp invalidation rate |
|---|---|---|---|
| Saved as PNG | 99.85% | 96.0% | 4.0% |
| JPEG quality 90 | 99.83% | 95.2% | 4.8% |
| JPEG quality 70 | 99.58% | 90.0% | 10.0% |
| JPEG quality 50 | 98.75% | 72.4% | 27.6% |
| Resized to 224 | 99.82% | 95.2% | 4.8% |
| Screenshot simulation | 99.71% | 94.2% | 5.8% |
| VAE round trip | 90.39% | 13.6% | 86.4% |
| Inpainting, 10% mask | 99.62% | 92.0% | 8.0% |
| Inpainting, 25% mask | 97.80% | 50.5% | 49.5% |
| Inpainting, 50% mask | 86.62% | 2.5% | 97.5% |

The stamp survives the screenshot simulation almost as well as a plain PNG (94.2% against 96.0% valid
verdicts). Blur on its own costs nothing; the largest single costs are the brightness and contrast
change and resizing to 80% and back (95.8% each). Adding JPEG quality 90 after the screenshot still
leaves 90.6% valid.

The VAE round trip breaks 86.4% of stamps, while the same resizing without the VAE leaves 96.8% valid,
so the loss comes from the VAE. Bit accuracy stays at 90.4%, far above guessing: the stamp is computed
at 128 pixels and enlarged, so it is a coarse pattern, and the VAE keeps coarse structure. Enough bits
change, however, for the checksum to reject most messages. Inpainting removes the stamp in proportion
to the redrawn area: 92.0% of stamps stay valid with a 10% mask, 50.5% with 25% and 2.5% with 50%.

In no test was a message that had been changed read as valid (0 "wrong but valid" in every row), and
no unstamped photo was read as carrying a valid stamp, either unchanged, after the screenshot
simulation or after the VAE round trip (0 of 500 each).

The answer to RQ2 is yes. The stamp survives the normal edits it was trained for (90% or more valid
verdicts up to JPEG quality 70 and after a screenshot), while a full AI regeneration breaks most stamps
and partial inpainting breaks them in proportion to the redrawn area. The gap is much larger in valid
verdicts (94.2% after a screenshot against 13.6% after the VAE) than in bit accuracy (99.7% against
90.4%), which is why both are reported. A missing or invalid stamp is therefore a strong signal that a
photo may have been regenerated, but not proof: strong JPEG compression also breaks stamps (27.6% at
quality 50), and 13.6% of stamps survived the VAE round trip.

When an inpainting tool returns its whole output instead of pasting the region back, every pixel has
passed through the model's VAE, and almost no stamps survive whatever the mask size (5.0%, 2.0% and
0.0% valid for 10%, 25% and 50% masks).

**Limits.** Cropping was not part of training and the stamp survives it poorly (73.59% of bits after a
centre crop on 128 pixel images, from the model's own test). The stamp is faintly visible in smooth
areas such as sky (see image quality).

### Image quality

The stamped PNG of each test photo is compared with the original at full size.

| Metric | Value on the 500 test photos |
|---|---|
| PSNR | 31.53 dB (median 31.58, lowest 25.86) |
| SSIM | 0.943 (median 0.948, lowest 0.690) |
| LPIPS (AlexNet) | 0.074 (median 0.070, highest 0.235) |

An SSIM of 0.94 and an LPIPS of 0.07 mean the stamp changes structure and appearance only slightly for
most photos. The stamp is not invisible everywhere, though: 13.4% of photos have an LPIPS above 0.1 and
6.6% an SSIM below 0.9, mostly photos with large smooth areas, where the pattern can be seen (figure
`results/figures/image_quality_worst.png`).

### Stamp on GenImage photos

M3 was trained on COCO photos but RQ3 stamps GenImage photos. On 300 Real and 300 AI photos from the
detectors' test split, bit accuracy after saving as PNG was 99.49% for Real and 99.80% for AI photos
(94.3% and 96.0% valid verdicts; PSNR 31.54 and 31.45 dB), and 99.41% and 99.45% after JPEG quality 90
(93.0% and 87.7% valid). The COCO-trained model therefore transfers to GenImage, and no fine-tuning was
needed. No unstamped GenImage photo was read as carrying a valid stamp.

---

## RQ3: does adding the watermark change the detector's prediction?

### Setup

We take 300 Real and 300 AI photos from M1's GenImage test split (seeded sample of the 1,300 test
photos: ADM and VQDM, the generators M1 was trained on, and Stable Diffusion v1.4, which it never saw).
The photos were read from the official GenImage archives and each matches the SHA-1 recorded in M1's
split files. Each photo is scored by M1, stamped with the verdict the pipeline would write, saved as
PNG and scored again. Both versions go through exactly the same steps before M1: the same file format
and M1's own preprocessing (the same JPEG re-encoding with the same per-image quality and the same
centre 224 crop). The stamp is computed at 128 pixels, enlarged and added to the full-size photo, so
the photo's size and detail do not change and the only difference between the two versions is the
stamp. Scoring the same photo twice gives identical scores. We report the mean change in P(AI) for
Real and for AI photos with 95% bootstrap intervals, the number of verdicts that change (Real, AI or
Uncertain), and accuracy and AUROC before and after stamping.

M1 was not retrained for this analysis. M2 was first evaluated with its original training, then
retrained with stamped training images added to the training split. Validation and test images stayed
clean. Fusion was then retrained using the updated M2 outputs. The 600 RQ3 photos use the Stable
Diffusion held-out fold: 444 are from `test_unseen` and 156 are from `test_seen`.

### Results

| Detector | Mean score change, Real photos | Verdict flips | AUROC before and after |
|---|---|---|---|
| M1 | +0.005 [−0.005, +0.015] | 51 of 600 (2 Real→AI, 2 AI→Real, 47 to or from Uncertain) | 0.754 / 0.748 |
| M2, original training | +0.0645 | 21.2% | 0.9339 / 0.9222 |
| M2, stamp-augmented training | +0.0225 | 17.2% | 0.9173 / 0.9094 |
| Late Fusion, original M2 | +0.0344 | 13.7% | 0.9030 / 0.8937 |
| Late Fusion, updated M2 | +0.0134 | 13.2% | 0.8880 / 0.8777 |

On average the stamp does not move M1's scores: both mean changes are close to zero and their 95%
intervals include zero, so the stamp does not push M1 towards "Real" or towards "AI". Accuracy
(67.5% → 67.2%) and AUROC (0.754 → 0.748) are almost unchanged. Individual scores do move, however:
the median change is 0.009, but one photo in ten moves by more than 0.15 and the largest change is
0.57. The size of these changes is the same for the three generators (mean absolute change 0.048 for
ADM, 0.049 for VQDM and 0.048 for Stable Diffusion v1.4). As a result 51 of the 600 verdicts change,
almost all between a firm verdict and Uncertain; a Real photo became AI in 2 cases and an AI photo
became Real in 2. At a single threshold of 0.5, 22 of 600 predictions flip.

The answer to RQ3 is different for the detectors. M1 adds noise to individual scores but shows no
clear systematic shift. The original M2 model shifted Real photos towards AI, with a mean change of
about +0.0645. This was reduced to +0.0225 after adding stamped training images. The Late Fusion
shift was reduced from +0.0344 to +0.0134 after the M2 update. Verdict flips also decreased for M2
and Late Fusion. The updated model therefore reduces stamp sensitivity, but it does not remove it
completely. AUROC also decreased slightly after the robustness update, so the change is a trade-off
between ordinary detector performance and stamp robustness.

For the three leave-one-generator-out M2 evaluation, the updated test-unseen AUROC values were
0.8821 for Stable Diffusion v1.4, 0.9548 for ADM, and 0.9498 for VQDM, with a mean of 0.9289.
The corresponding Late Fusion AUROC values were 0.8575, 0.9447, and 0.9536, with a mean of 0.9186.
These results use the same fold definitions as M1. They are reported separately from the clean-data
comparison because RQ3 is a robustness test.

---

## The decode, detect and stamp pipeline

The pipeline first reads the stamp. If the checksum passes, it returns the stored verdict and skips
the detectors. Otherwise it runs M1, M2 and Late Fusion, then turns P(AI) into Real, AI or Uncertain
with thresholds fitted on clean validation data. A valid stamp only means that our
detector gave that verdict when the photo was stamped: the stamp is not signed, so it is not proof
that the photo is authentic.

On 200 Real and 200 AI GenImage test photos, no unstamped photo was mistaken for a stamped one (0 false
skips). 93 of the 200 AI photos (46.5%) were stamped as Real, the system metric from the proposal; 13
Real photos were stamped as AI and 6.8% of photos were Uncertain. Almost all AI photos stamped as Real
come from Stable Diffusion v1.4 (91 of 148, against 2 of 52 for ADM and VQDM), the generator M1 never
saw in training; this is a limit of M1, not of the stamp. When a stamped output was fed back in, 93.5%
were answered from the stamp (99.5% of those with the same verdict as the first pass), 88.5% after JPEG
quality 90 (98.8% the same verdict), and 17.8% after a VAE round trip, where the stamp breaks and the
detectors run again (82.0% gave the same verdict as before). The pipeline took 0.03 s per photo on a
Tesla T4.
