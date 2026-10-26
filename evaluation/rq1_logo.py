"""RQ1: leave-one-generator-out tables for M1, M2, Fusion and baselines (owner: Workstream B).

Reads predictions.csv from each run in experiments/, never retrains. Outputs
reports/tables/rq1.csv with in-distribution and held-out accuracy + AUROC per fold,
plus the JPEG 65/30 degradation track (Workstream A).
"""
