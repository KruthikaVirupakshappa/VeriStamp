# experiments/

One folder per run, created by `common.runlog.start_run()`, containing `config.yaml`,
`meta.json`, `metrics.json` and `predictions.csv`. Run folders are git-ignored.
Upload important ones to the shared Drive.

`runs_log.csv` **is** committed: one row per run, including failed runs.
Every number in the report must trace back to a `run_id` in this file.
