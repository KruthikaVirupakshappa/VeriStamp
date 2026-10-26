"""Train M3 on COCO at 128x128 (owner: Workstream C).

    python -m training.train_watermark --config configs/m3.yaml

TODO: adapt the public HiDDeN training loop; log bit accuracy (no edit + JPEG) and PSNR
every epoch; the interim target is >95% bit accuracy after JPEG.
"""
