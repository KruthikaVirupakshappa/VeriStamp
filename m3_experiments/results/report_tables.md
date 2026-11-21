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
| M1 | +0.0049 [-0.0048, +0.0147] | -0.0076 [-0.0204, +0.0057] | 51 of 600 (2 Real to AI, 2 AI to Real, 47 to or from Uncertain) | 67.50% / 67.17% |
| M2 | not run yet | not run yet | not run yet | not run yet |
| Fusion | not run yet | not run yet | not run yet | not run yet |

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
| AI photos stamped as Real | 93 of 200 (46.5%) |
| Real photos stamped as AI | 13 |
| Uncertain share | 6.8% |
| mean seconds per photo (first pass) | 0.0290 |
| returned as PNG: answered from stamp | 93.5% |
| returned as PNG: same verdict as first pass | 99.5% |
| returned after JPEG 90: answered from stamp | 88.5% |
| returned after JPEG 90: same verdict as first pass | 98.8% |
| returned after VAE round trip: answered from stamp | 17.8% |
| returned after VAE round trip: same verdict as first pass | 82.0% |

| class | Real | Uncertain | AI |
|---|---|---|---|
| AI | 93 | 15 | 92 |
| Real | 175 | 12 | 13 |

