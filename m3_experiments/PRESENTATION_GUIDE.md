# M3 presentation guide

Everything needed to present M3 (the watermark) in the final presentation: the story, a slide-by-slide
outline with what to say, every key number, the figures to show, and answers to likely questions.
All numbers come from the Kaggle runs on a Tesla T4 (20 and 22 November 2026); the full tables are in
`results/report_tables.md` and the full text in `REPORT_DRAFT.md`.

---

## 1. The story in one minute

VeriStamp checks whether a photo is real or AI-made (M1, M2 and Late Fusion), then hides that verdict
invisibly inside the photo with M3, so anyone can read it back later without running the detectors.

We asked two questions about M3:

- **RQ2: does the stamp survive normal edits better than AI edits?** Yes. It survives a screenshot
  (94% of stamps stay valid) but AI regeneration breaks most stamps (only 14% survive). A missing
  stamp is therefore a strong hint that a photo was regenerated.
- **RQ3: does adding the stamp change the detectors' answers?** It barely changes M1, but it nudges
  M2 towards "AI". That is the opposite of what we predicted, and the likely reason is that the stamp
  is a coarse pattern, which CLIP (M2) notices and M1's high-pass filter removes.

In the full system, Late Fusion of M1 and M2 halves the number of AI photos stamped as "Real"
(22.5% against 46.5% with M1 alone), and no unstamped photo was ever mistaken for a stamped one.

---

## 2. Suggested slides (about 8 to 10 minutes for the M3 part)

### Slide 1. What M3 does (1 min)

**Show:** one photo before and after stamping (`results/figures/rq2_screenshot_example.png`, first two
panels), plus the 32-bit message as a small diagram:

| Verdict | Confidence | Detector | Image tag | Checksum |
|---|---|---|---|---|
| 2 bits | 4 bits | 4 bits | 14 bits | 8 bits (CRC-8) |

**Say:**
- M3 hides a 32-bit message in the photo: the verdict (Real, AI or Uncertain), the confidence, which
  detector decided, an image tag, and a checksum.
- The checksum lets us tell a real message from noise: if bits are damaged, the stamp counts as invalid
  instead of giving a wrong verdict.
- The model works on a 128 × 128 copy of the photo: it computes a faint pattern, enlarges it and adds it
  to the full-size photo. To read it, the photo is shrunk back to 128 × 128. (Remember this: it explains
  both main findings.)

### Slide 2. How we tested it (1 min)

**Say:**
- **Test data only:** 500 COCO photos M3 never saw in training, at full size, and 1,300 GenImage test
  photos for the detector experiments.
- **Verified inputs:** the model reproduces the training team's numbers exactly; every GenImage photo
  was taken from the official archives and matches the SHA-1 fingerprint in M1's split files.
- **Three measures:** bit accuracy (share of the 32 bits read correctly; 50% = guessing), valid verdict
  (checksum passes and the verdict is right) and invalidation rate (checksum fails). We report both bit
  accuracy and valid verdict because a few wrong bits are enough to break the checksum.
- **Reproducible:** fixed seeds, all runs on Kaggle; repeated runs gave identical numbers.

### Slide 3. Is the stamp visible? (45 s)

**Show:** `results/figures/image_quality_worst.png` (original, stamped, difference × 10).

| Measure | Result | Meaning |
|---|---|---|
| PSNR | 31.5 dB | small pixel differences (above 30 dB is generally good) |
| SSIM | 0.943 | structure almost unchanged (1 = identical) |
| LPIPS | 0.074 | small perceived difference (below 0.1 is small) |

**Say:** hard to see for most photos, but honest: 13% of photos have LPIPS above 0.1, mostly photos with
large smooth areas such as sky, where the pattern can be faintly seen.

### Slide 4. RQ2: normal edits vs AI edits (2 min, the main slide)

**Show:** this table (or a bar chart of the "Valid verdict" column).

| Edit | Bit accuracy | Valid verdict |
|---|---|---|
| Saved as PNG (baseline) | 99.85% | 96.0% |
| JPEG quality 90 | 99.83% | 95.2% |
| JPEG quality 70 | 99.58% | 90.0% |
| JPEG quality 50 | 98.75% | 72.4% |
| Screenshot simulation | 99.71% | 94.2% |
| VAE round trip (AI regeneration) | 90.39% | **13.6%** |
| Inpainting, 10% redrawn | 99.62% | 92.0% |
| Inpainting, 25% redrawn | 97.80% | 50.5% |
| Inpainting, 50% redrawn | 86.62% | **2.5%** |

**Say:**
- **Normal edits are survived:** a screenshot (shrink to 80%, slight blur, brightness/contrast change,
  enlarge back) keeps 94% valid, almost as many as a plain PNG.
- **AI regeneration breaks the stamp:** passing the photo through Stable Diffusion's encoder and decoder
  leaves only 14% valid. A control with the same resizing but no VAE keeps 97%, so the damage is the VAE's.
- **Inpainting breaks it in proportion to the area redrawn:** 92% valid at 10%, 51% at 25%, 3% at 50%.
- **Note the two columns:** bit accuracy after the VAE is still 90%, yet only 14% of messages are valid.
  The stamp is coarse, so the VAE keeps most of it, but enough bits flip to break the checksum.
- **Safety:** no damaged message ever passed the checksum with the wrong content, and no unstamped photo
  was read as stamped.

**Conclusion line:** "A missing stamp is a strong hint of AI regeneration, but not proof: heavy JPEG also
breaks stamps, and 14% survived the VAE."

### Slide 5. Inpainting in a picture (30 s, optional)

**Show:** `results/figures/rq2_inpainting.png` (bit accuracy and valid verdict against mask size).

**Say:** the redrawn region was pasted back into the photo, as a photo editor does. If a tool returns
the model's whole output instead, every pixel passes through the VAE and almost no stamps survive
(5% or less at any size).

### Slide 6. Does it work on the detectors' photos? (30 s)

**Say:** M3 was trained on COCO, but the detectors use GenImage, which includes smoother AI images. On
300 Real and 300 AI GenImage photos, 99.5% and 99.8% of bits were read correctly after saving, above
the 95% bar. So no extra training was needed.

### Slide 7. RQ3: does the stamp change the detectors? (2 min, the second main slide)

**Show:** `results/figures/rq3_score_change.png` (histograms of P(AI) after minus before) and this table.

| Detector | Mean change, Real photos | Mean change, AI photos | Verdicts changed | Accuracy before → after |
|---|---|---|---|---|
| M1 (high-pass + FFT ResNet) | +0.004 | −0.013 | 53 of 600 | 67.5% → 66.7% |
| M2 (CLIP) | **+0.063** | **+0.053** | 118 of 600 | 82.5% → 83.2% |
| Late Fusion | +0.034 | +0.020 | 72 of 600 | 75.2% → 75.2% |

**Say:**
- **Setup:** 600 GenImage test photos; each is scored, stamped, and scored again with identical
  preprocessing, so the only difference is the stamp.
- **We verified we used the real detectors:** before stamping, each reproduced its owner's score for all
  600 photos (largest difference 0.00001).
- **M1 barely reacts:** its changes are centred on zero.
- **M2 is pushed towards "AI":** P(AI) rises for 71% of photos, by about 0.06 on average (95% range
  +0.048 to +0.080 on Real photos, clearly above zero). On Real photos, "Real" verdicts drop from 265 to
  241, mostly into Uncertain; only 1 more becomes "AI".
- **Late Fusion gets half of M2's push** and no verdict flips between Real and AI.

### Slide 8. Why the result is the opposite of our prediction (1 min)

**Say:**
- The proposal expected the frequency-based M1 to react more, because we assumed the stamp was fine
  high-frequency noise.
- In fact the stamp is made at 128 pixels and enlarged, so it is a **coarse** pattern. M1's high-pass
  filter removes coarse patterns; CLIP's features respond to exactly that scale.
- The VAE result supports this: most of the stamp's bits survive regeneration because the VAE keeps
  coarse structure.
- **Does it matter in practice?** Not for the verdict a photo gets: the pipeline scores a photo before
  stamping it and never re-scores a photo with a valid stamp. It matters only if a stamped photo is
  scored again, for example after an edit breaks the stamp.

### Slide 9. The full system (1 min)

**Show:** the flow: photo → read stamp → (valid: return stored verdict) / (invalid: M1 + M2 → Late Fusion
→ Real / Uncertain / AI → stamp) → stamped photo.

| System result (400 GenImage photos) | Value |
|---|---|
| Unstamped photos mistaken for stamped | 0 |
| AI photos stamped as Real | 22.5% (46.5% with M1 alone) |
| Real photos stamped as AI | 5 of 200 |
| Uncertain | 24.8% |
| Stamped photo sent back: answered from the stamp | 94.8% (all with the same verdict) |
| … after JPEG 90 | 90.2% |
| … after AI regeneration | 15.8% (the detectors run again, as intended) |
| Time per photo (Tesla T4) | 0.05 s |

**Say:** the stamp is not signed: a valid stamp means "our detectors said this when the photo was
stamped", not "this photo is authentic". Someone with the model could forge a stamp.

### Slide 10. Limitations and future work (45 s)

- One seed; 600 photos for RQ3 and 200 for inpainting.
- The stamp is weak against cropping (cropping was not in training) and faintly visible in smooth areas.
- The stamp nudges M2 towards "AI"; training M3 with the detectors in the loop, or a finer stamp, could
  fix this.
- M1's Uncertain thresholds are placeholders (0.35 / 0.65); M2 and Late Fusion use fitted ones.
- No cryptographic signature: future work could sign the message so stamps cannot be forged.

---

## 3. Figures to use (all in `results/figures/`)

| Figure | Slide | What it shows |
|---|---|---|
| `rq2_screenshot_example.png` | 1 | original, stamped, after the screenshot simulation |
| `image_quality_worst.png` | 3 | the photos where the stamp is most visible, with the difference × 10 |
| `rq2_vae_example.png` | 4 | a stamped photo before and after the VAE round trip |
| `rq2_inpainting.png` | 5 | stamp survival against the area redrawn |
| `rq3_score_change.png` | 7 | how much each detector's score moves when a photo is stamped |

---

## 4. Likely questions and answers

| Question | Answer |
|---|---|
| Why 32 bits? | Enough for a verdict, confidence, detector, image tag and an 8-bit checksum; a longer message is harder to hide and to read back. |
| What does the checksum buy you? | It turns damage into "invalid" instead of a wrong verdict. In every test, no damaged message passed the checksum with the wrong content. |
| Why both bit accuracy and valid verdict? | A few wrong bits break the checksum. After the VAE, 90% of bits survive but only 14% of messages are valid. |
| How do you know the GenImage photos are the right ones? | They come from the official archives, and each matches the SHA-1 fingerprint in M1's split files. |
| How do you know RQ3 used the real detectors? | Before stamping, each reproduced its owner's score for all 600 photos (largest difference 0.00001), and their AUROC matched the reported values (M2 0.909 against 0.906, Late Fusion 0.874 against 0.878). |
| Why only 200 photos for inpainting? | It is the handover guide's minimum, and inpainting is slow (about an hour on a T4). The subset was a fixed random sample. |
| Why Stable Diffusion 1.5 inpainting, not 2? | The version 2 model was removed from Hugging Face; 1.5 inpainting is the standard replacement. |
| Why does M2 react more than M1? | The stamp is coarse (made at 128 px and enlarged). M1's high-pass filter removes it; CLIP responds to that scale. |
| Does that make the system unreliable? | No for the verdict a photo receives: photos are scored before stamping, and stamped photos are not re-scored. Only re-scoring a stamped photo is affected. |
| Why does Late Fusion help so much in the pipeline? | M1 struggles on Stable Diffusion images it never saw in training; M2 handles them well, and averaging the two cuts AI-stamped-as-Real from 46.5% to 22.5%. |
| Can someone remove the stamp? | Yes, by AI regeneration, heavy inpainting, heavy JPEG or cropping. The system treats a missing stamp as "unknown" and runs the detectors again, so removal does not create a false "Real". |
| Can someone fake a stamp? | Yes, with the model, because the message is not signed. That is why we say "valid stamp with stored verdict", never "verified authentic". |
| Is it reproducible? | Yes: fixed seeds, test data only, all notebooks with their outputs in `executed/`, and repeated Kaggle runs gave identical numbers. |

---

## 5. Words to use (and avoid)

- Say **"stamp"** or **"watermark"** for M3's hidden message, and **"valid stamp"** when the checksum
  passes.
- Say **"valid stamp with stored verdict"**, never **"verified"** or **"authentic"**.
- Say **"regenerated"** for the VAE round trip and **"redrawn"** for inpainting.
- **P(AI)** is the detector's probability that the photo is AI-made; **AUROC** measures how well a
  detector ranks AI photos above real ones (0.5 = guessing, 1 = perfect).

---

## 6. Who did what (for the credits slide; check the names with the team)

- **M3 model and training:**  Rishitha`m3_watermark/`, notebooks 1 to 4.
- **M3 evaluation (RQ2, image quality, GenImage check, RQ3, pipeline) and report sections:**
  `m3_experiments/`.
- **M1:** Kruthika. **M2 and Fusion:** Supriya.
