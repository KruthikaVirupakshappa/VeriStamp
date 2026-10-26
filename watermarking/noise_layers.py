"""Differentiable training-time distortions for M3 (owner: Workstream C).

TODO: identity, JPEG approximation (JPEG-Mask / straight-through real JPEG),
resize, crop, screenshot simulation. Each is an nn.Module; during training one
is sampled per batch. Real JPEG has no useful gradient, so for the straight-through
version use:  y = x + (jpeg(x) - x).detach()
"""
