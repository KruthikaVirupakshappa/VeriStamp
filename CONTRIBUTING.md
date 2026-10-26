# How we work

## Branches and pull requests
- `main` must always pass `pytest` and `python -m tests.integration`. Nobody pushes to it directly.
- One branch per task, prefixed by workstream: `a/splits`, `b/m1-fft`, `c/noise-layer`, `d/rq3-harness`.
- Open a pull request early (a draft PR is fine). Each PR is reviewed by **one person from a different
  workstream**, which spreads knowledge of the codebase.
- Pull `main` into your branch at least twice a week.

```bash
git checkout main && git pull
git checkout -b b/m2-clip-cache
# ... work, commit small steps ...
git push -u origin b/m2-clip-cache       # then open a PR on GitHub
```

## Never commit
Images, datasets, checkpoints (`*.pt`), run folders, notebook outputs, API keys.
Clear notebook outputs before committing (Kernel → Restart & Clear Output).

## Shared files: change only with a heads-up in the team chat
`configs/base.yaml`, `preprocessing/transforms.py`, `data/splits/*.csv`, `watermarking/payload.py`,
`models/thresholds.py`, `evaluation/metrics.py`, `docs/interfaces.md`.
Changing any of these can invalidate other people's results.

## Before opening a PR
- [ ] `pytest` passes
- [ ] `python -m tests.integration --n 20` passes
- [ ] New code that others call has a docstring stating input and output shapes
- [ ] Any experiment you ran has a row in `experiments/runs_log.csv`

## Weekly rhythm
- **Integration check:** run `python -m tests.integration` on `main` and fix breakages first.
- **Progress meeting:** each workstream shows the items listed in the guide (Section 15),
  then updates its row in the learning dashboard.
