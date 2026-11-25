# Code for the M3 tables and figures of the report

Everything here reads the result files in `../results`, so the numbers in the report can be regenerated
instead of retyped.

| Report item | Produced by | Source of the numbers |
|---|---|---|
| Table 5, training stages | the test cells at the end of notebooks 1 to 3 in `../../m3_watermark` | `test_summary.csv` in the results zips on the team Drive (not in this repo) |
| Table 6, RQ2 edits | `make_tables.py` | `../results/setup_check_meta.json`, `rq2_screenshot_per_photo.csv`, `rq2_vae_per_photo.csv`, `rq2_inpainting_per_photo.csv` |
| Table 7, RQ3 detectors | `make_tables.py` | `../results/rq3_detectors_summary.csv` |
| Fig. 1, pipeline diagram | `make_figures.py` | the pipeline steps; thresholds from `../results/pipeline_meta.json` |
| Fig. 2, bit accuracy and valid verdict by edit | `make_figures.py` | the numbers of Table 6 |
| Fig. 3, JPEG at full size versus 128 px | `make_figures.py` | full size: Table 6; 128 px: `test_summary.csv` of `results_3_final_model.zip` (typed in the script) |
| Fig. 4, change in P(AI) when a photo is stamped | `make_figures.py` | `../results/rq3_detectors_per_photo.csv` |

The experiments themselves (notebooks 5 to 13) are in `..`; their executed copies are in `../executed`.

## How to use
Run from this folder:

    python make_tables.py       # prints the LaTeX rows of Tables 6 and 7
    python make_figures.py      # writes the four PDFs to ./figs  (needs matplotlib and numpy)

`make_tables.py` needs only the Python standard library. The four PDFs are committed in `figs/` so they can
be downloaded and uploaded to Overleaf without running anything.

Notes
- The report's tables are plain LaTeX. Paste the printed rows into the matching `tabular`.
- Table 7 is computed from the full-precision values. An earlier version of the report had three confidence
  interval bounds that were rounded twice (M1 on AI photos upper bound, M2 on Real photos upper bound, Late Fusion
  on Real photos lower bound); this script gives the exact values.
- The 128 px numbers in Fig. 3 are not in this repo, so they are typed in `make_figures.py` with their source noted.
