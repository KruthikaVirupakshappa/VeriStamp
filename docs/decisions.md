# Team decisions

Each entry: the question, the current default (already in the code/configs), and the decision once made.
When you decide, replace "OPEN" with the decision and the date, and update the config if needed.

| # | Question | Current default | Status |
|---|---|---|---|
| 1 | Which 3 GenImage generators? | `sdv14`, `adm`, `biggan`: one per family (latent diffusion, pixel diffusion, GAN) | OPEN |
| 2 | Detector image size | shorter side → 256, centre crop 224 | OPEN |
| 3 | JPEG re-encode range (bias control) | quality 75–95, deterministic per image | OPEN |
| 4 | How does M3 (128×128) stamp a 224×224 image? | compute residual at 128, upsample, add at 224; RQ3 always includes a resize-only control | OPEN |
| 5 | CLIP model | ViT-B/16 first; ViT-L/14 (proposal) once the pipeline works | OPEN |
| 6 | M1 input | 4 channels: high-pass residual (3) + FFT log-magnitude (1), adapted conv1 | OPEN |
| 7 | Payload layout | 32 bits: verdict 2 + P(AI) bucket 4 + detector id 4 + image tag 14 + CRC-8 8 | OPEN |
| 8 | What makes a stamp invalid? | CRC mismatch or reserved verdict code (`payload.decode`) | OPEN |
| 9 | Edits and strengths for RQ2 | `configs/rq2_edits.yaml` | OPEN |
| 10 | Uncertain threshold rule | on validation: AI-called-Real ≤ 2% and Uncertain ≤ 20%, best accuracy on decided | OPEN |
| 11 | Fusion | early fusion (standardise + concat) main; late fusion (average) as ablation | OPEN |
| 12 | Minimum viable scope we commit to | the "Minimum viable project" column in the guide, Section 10 | OPEN |
| 13 | Workstream owners | A: ?, B: ?, C: ?, D: ? | OPEN |
| 14 | Compute plan | who uses Colab / Kaggle / SJSU HPC; shared GPU-hour tally vs the 50–100 h budget | OPEN |
