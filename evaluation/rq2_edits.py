"""RQ2: bit accuracy and stamp invalidation per edit (owner: Workstream C).

For each image in the COCO test set: random payload -> embed -> each edit in
configs/rq2_edits.yaml -> extract -> payload.decode. Outputs reports/tables/rq2.csv
and PSNR / SSIM / LPIPS of stamped vs cover (evaluation.metrics).
"""
