# Report sections: RQ2 edits, image quality, RQ3 and the pipeline

Text for the report sections the handover guide assigns to M3 (tasks 8 and 11), filled in from the
Kaggle runs of 20 November 2026 (RQ2, image quality, GenImage check) and 22 November 2026 (RQ3 and the
pipeline with M1, M2 and Late Fusion), on a Tesla T4. Every number below is in `results/report_tables.md`
or the per-photo CSVs next to it; the run notebooks are in `executed/`. One input is still a
placeholder: M1's Uncertain thresholds (0.35 and 0.65), which only affect M1's verdict counts. M2 and
Late Fusion use the thresholds their owners fitted on validation.

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

We take 300 Real and 300 AI photos from the detectors' GenImage test split for the fold in which
Stable Diffusion v1.4 is held out (ADM and VQDM seen, SD v1.4 unseen; seeded sample of the 1,300 test
photos). The photos were read from the official GenImage archives, and each matches the SHA-1 recorded
in M1's split files. Each photo is scored by M1, M2 and Late Fusion ((M1 + M2) / 2), stamped with the
verdict the pipeline would write (Late Fusion's), saved as PNG and scored again. Both versions go
through exactly the same steps before each detector: the same file format and each detector's own
preprocessing (for both, a JPEG re-encoding at the same per-image quality, a centre 256 tile and a
centre 224 crop). The stamp is computed at 128 pixels, enlarged and added to the full-size photo, so
the photo's size and detail do not change and the only difference between the two versions is the
stamp.

Before stamping, every detector reproduces its owner's own score for each of the 600 photos (largest
difference 0.00001, correlation 1.0; M1 against the scores in the fusion files, M2 and Late Fusion
against their prediction files), and its AUROC on the unseen SD v1.4 photos matches the reported
value (M1 0.684 here against 0.712, M2 0.909 against 0.906, Late Fusion 0.874 against 0.878). RQ3
therefore tests the team's models exactly as their owners ran them.

### Results

| Detector | Score change, real photos | Score change, AI photos | Verdict flips | Accuracy before and after |
|---|---|---|---|---|
| M1 | +0.004 [−0.006, +0.014] | −0.013 [−0.025, −0.002] | 53 of 600 (0 Real→AI, 1 AI→Real, 52 to or from Uncertain) | 67.5% / 66.7% |
| M2 | +0.063 [+0.048, +0.080] | +0.053 [+0.031, +0.075] | 118 of 600 (3 Real→AI, 1 AI→Real, 114 to or from Uncertain) | 82.5% / 83.2% |
| Late Fusion | +0.034 [+0.024, +0.044] | +0.020 [+0.008, +0.033] | 72 of 600 (all to or from Uncertain) | 75.2% / 75.2% |

Score changes are mean P(AI) after minus before, with 95% bootstrap intervals; accuracy is at 0.5.

M1 is barely affected. Its changes are centred on zero (P(AI) rises for 51% of photos, median change
0.000) and its mean change for Real photos is not different from zero; for AI photos it is a very small
decrease (−0.013). Individual scores move by a median of 0.009, and its accuracy (67.5% → 66.7%) and
AUROC (0.754 → 0.740) hardly change.

M2 reacts more, and in one direction: stamping raises P(AI). It rises for 71% of photos, by 0.063 on
average for Real photos and 0.053 for AI photos, with both intervals well above zero; one photo in ten
moves by more than 0.30. The shift is strongest on Stable Diffusion photos (+0.069) and smaller on ADM
(+0.028) and VQDM (+0.022). On Real photos it moves verdicts towards caution: 241 of 300 are called Real
after stamping against 265 before, 50 Uncertain against 27, and only 9 AI against 8. On AI photos it
helps (152 called AI against 129). Accuracy is unchanged (82.5% → 83.2%), but the ranking of photos is
slightly worse (AUROC 0.930 → 0.916). Late Fusion averages the two and inherits about half of M2's
shift (+0.034 for Real photos; Real verdicts on Real photos 257 → 245); no Late Fusion verdict flipped
between Real and AI.

The answer to RQ3 is that the stamp barely changes M1's predictions but moves M2's towards AI. This is
the opposite of the proposal's expectation that the frequency-based M1 would react more than the
CLIP-based M2. A likely reason fits the RQ2 results: the stamp is computed at 128 pixels and enlarged,
so it is a coarse pattern rather than fine high-frequency noise. M1's high-pass filter removes most of
such a pattern, while CLIP's image features respond to structure at that scale (the same coarseness
lets most of the stamp's bits survive the VAE round trip).

In the pipeline this does not affect the verdict a photo receives, because the detectors score the
photo before it is stamped, and a photo with a valid stamp is not scored again. It matters when a
stamped photo is scored afterwards, for example when an edit breaks the stamp and the detectors run
again: M2 and Late Fusion will then lean slightly towards AI and Uncertain on real photos. Training
M3 with the detectors in the loop, or with a weaker or finer stamp, could reduce this.

## The decode, detect and stamp pipeline

The pipeline first reads the stamp. If the checksum passes, it returns the stored verdict and skips
the detectors. Otherwise it runs M1 and M2, combines them with Late Fusion ((M1 + M2) / 2), turns the
fused P(AI) into Real, AI or Uncertain with Late Fusion's validation thresholds (0.325 and 0.754), and
stamps that verdict into the photo with M3. A valid stamp only means that our detectors gave that
verdict when the photo was stamped: the stamp is not signed, so it is not proof that the photo is
authentic.

On 200 Real and 200 AI GenImage test photos, no unstamped photo was mistaken for a stamped one (0 false
skips). 45 of the 200 AI photos (22.5%) were stamped as Real, the system metric from the proposal,
against 46.5% when M1 alone made the verdict; 5 Real photos were stamped as AI, and 24.8% of photos were
Uncertain. When a stamped output was fed back in, 94.8% were answered from the stamp (all with the same
verdict as the first pass), 90.2% after JPEG quality 90 (99.0% the same verdict), and 15.8% after a VAE
round trip, where the stamp breaks and the detectors run again (70.2% gave the same verdict as before).
The pipeline took 0.05 s per photo on a Tesla T4.
