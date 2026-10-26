# Dataset card (owner: Workstream A — fill this in)

## Sources
| Dataset | Use | Where from | Licence | Date downloaded |
|---|---|---|---|---|
| GenImage: `<generator 1>` + its real images | detector train/val/test | | | |
| GenImage: `<generator 2>` | | | | |
| GenImage: `<generator 3>` | | | | |
| COCO (subset, 128×128) | M3 training + RQ2 edit suite | | | |

## Expected layout
```
data/raw/genimage/<generator>/ai/       AI images
data/raw/genimage/<generator>/nature/   matching real images
data/raw/coco/                          COCO images
```

## Counts (from `python -m preprocessing.build_splits`)
_paste the printed table here_

## Preprocessing (bias control)
Every image, real or AI, goes through `preprocessing/transforms.py::preprocess`:
resize the shorter side to 256, centre-crop to 224, then JPEG re-encode with a quality drawn
deterministically per image from `[75, 95]`. Record here anything that differs between
the classes before preprocessing (formats, sizes), and how you checked it.

## Known issues / decisions
-
