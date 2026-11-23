## RQ2: edits

| Edit | Bit accuracy | Valid verdict | Stamp invalidation rate |
|---|---|---|---|
| Saved as PNG | 99.85% | 96.0% | 4.00% |
| JPEG quality 90 | 99.83% | 95.2% | 4.80% |
| JPEG quality 70 | 99.58% | 90.0% | 10.00% |
| JPEG quality 50 | 98.75% | 72.4% | 27.60% |
| Resized to 224 | 99.82% | 95.2% | 4.80% |
| Screenshot simulation | 99.71% | 94.20% | 5.80% |
| VAE round trip | 90.39% | 13.60% | 86.40% |
| Inpainting, 10% mask | 99.62% | 92.00% | 8.00% |
| Inpainting, 25% mask | 97.80% | 50.50% | 49.50% |
| Inpainting, 50% mask | 86.62% | 2.50% | 97.50% |

The first five rows repeat notebook 4 (notebook 5; the same as section 1 of the guide within 0.2 points).

## Image quality

| Metric | Value on the 500 test photos |
|---|---|
| PSNR | 31.53 dB |
| SSIM | 0.9428 |
| LPIPS | 0.0737 |

## RQ3: detectors

| Detector | Score change, real photos | Score change, AI photos | Verdict flips | Accuracy before / after |
|---|---|---|---|---|
| M1 | +0.0043 [-0.0061, +0.0144] | -0.0130 [-0.0249, -0.0015] | 53 of 600 (0 Real to AI, 1 AI to Real, 52 to or from Uncertain) | 67.50% / 66.67% |
| M2 | +0.0626 [+0.0478, +0.0795] | +0.0531 [+0.0314, +0.0748] | 118 of 600 (3 Real to AI, 1 AI to Real, 114 to or from Uncertain) | 82.50% / 83.17% |
| Late Fusion | +0.0335 [+0.0235, +0.0440] | +0.0200 [+0.0076, +0.0327] | 72 of 600 (0 Real to AI, 0 AI to Real, 72 to or from Uncertain) | 75.17% / 75.17% |

### Stamp on GenImage photos

| class | edit | n_photos | bit_accuracy | valid_verdict | invalidation_rate | all_32_bits_correct | wrong_but_valid | psnr | ssim |
|---|---|---|---|---|---|---|---|---|---|
| Real | Saved as PNG | 300 | 99.49% | 94.33% | 5.67% | 94.33% | 0 | 31.5408 | 0.9399 |
| Real | JPEG quality 90 | 300 | 99.41% | 93.00% | 7.00% | 93.00% | 0 | 31.5408 | 0.9399 |
| AI | Saved as PNG | 300 | 99.80% | 96.00% | 4.00% | 96.00% | 0 | 31.4535 | 0.9358 |
| AI | JPEG quality 90 | 300 | 99.45% | 87.67% | 12.33% | 87.67% | 0 | 31.4535 | 0.9358 |

### Pipeline

| metric | value |
|---|---|
| photos | 400 |
| false skips (unstamped photo read as stamped) | 0 |
| AI photos stamped as Real | 45 of 200 (22.5%) |
| Real photos stamped as AI | 5 |
| Uncertain share | 24.8% |
| mean seconds per photo (first pass) | 0.0490 |
| returned as PNG: answered from stamp | 94.8% |
| returned as PNG: same verdict as first pass | 100.0% |
| returned after JPEG 90: answered from stamp | 90.2% |
| returned after JPEG 90: same verdict as first pass | 99.0% |
| returned after VAE round trip: answered from stamp | 15.8% |
| returned after VAE round trip: same verdict as first pass | 70.2% |

| class | Real | Uncertain | AI |
|---|---|---|---|
| AI | 45 | 79 | 76 |
| Real | 175 | 20 | 5 |

