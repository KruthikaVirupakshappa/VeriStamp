"""RQ3: does stamping change the detector? (owner: Workstream D)

For each test image and each detector (M1, M2, Fusion), three arms:
  (a) original, (b) control: the same resize/re-encode path as stamping but NO watermark,
  (c) stamped.
Report the watermark effect as (c) - (b): mean score change, flip rate, Real->AI flips,
AI stamped as Real, decoded-verdict agreement. Outputs reports/tables/rq3.csv.
"""
