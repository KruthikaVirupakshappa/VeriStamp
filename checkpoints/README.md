# checkpoints/

Weights are **not** stored in git. They live in the team's shared Drive folder.

- File name: `{model}_{fold}_{YYYYMMDD}_{gitsha}.pt`. Example: `m1_resnet_fold0_20261030_a1b2c3d.pt`
- Save a dict: `{"state_dict", "config", "epoch", "val_metrics"}`
- Add one row to `REGISTRY.csv` for every checkpoint you upload
- Set `frozen_for_rq3 = yes` on the one checkpoint per model used for RQ3, and never overwrite it
