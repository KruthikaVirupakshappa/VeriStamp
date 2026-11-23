"""Detector wrappers for RQ3 (task 9) and the pipeline (task 10).

Every detector here is a callable: detector(photo: PIL.Image, image_id: str) -> {'p_ai': float,
'features': np.ndarray or None}. image_id must be the same for a photo before and after stamping,
because M1's preprocessing picks its JPEG quality from it.

  DummyDetector  works now. Its score comes from the photo's fine detail, so it reacts a little to
                 the stamp, like a frequency detector would. It means nothing about Real vs AI.
  M1Detector     the real M1 (Kruthika), loaded from the M1 notebook's own modules and best.pt.
                 Its preprocessing is M1's own preprocess_native + centre 224 crop.
                 Confirm the paths and steps with Kruthika before using its numbers.
  M2Detector     the real M2 (Supriya): frozen CLIP ViT-B/16 + MLP head, with M2's own preprocessing.
  late_fusion    (M1 + M2) / 2, the team's Late Fusion (README_FUSION.md), the stronger fusion on unseen
                 generators (mean AUROC 0.925 against 0.840 for Early Fusion).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


class DummyDetector:
    name = 'dummy'

    def __call__(self, photo, image_id=''):
        g = np.asarray(photo.convert('L'), dtype=np.float32) / 255
        energy = (np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=0)).mean()) / 2
        p = float(1 / (1 + np.exp(-(energy - 0.03) * 60)))
        return {'p_ai': p, 'features': np.array([energy], dtype=np.float32)}


class M1Detector:
    """M1 from notebooks/M1_final_n500.ipynb: ResNet-50 on high-pass + FFT input."""
    name = 'm1_resnet'

    def __init__(self, vslib_dir, checkpoint, device=None):
        import torch
        vslib_dir = str(Path(vslib_dir))
        if vslib_dir not in sys.path:
            sys.path.insert(0, vslib_dir)
        import vs_models    # written by the M1 notebook (Section 2)
        import vs_native    # preprocess_native lives here
        self.torch, self.vs_native = torch, vs_native
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        ck = torch.load(checkpoint, map_location=self.device, weights_only=False)
        cfg = dict(ck['config'])
        cfg['model'] = {**cfg['model'], 'pretrained': False}       # weights come from the checkpoint
        self.model = vs_models.build_detector(ck.get('input_mode', 'hp_fft'), cfg).to(self.device)
        self.model.load_state_dict(ck['model'])
        self.model.eval()
        self.pcfg, self.seed = cfg['preprocess'], int(cfg.get('seed', 42))
        self.crop = int(self.pcfg.get('crop_size', 224))
        self.align = int(self.pcfg.get('align', 8))

    def preprocess(self, photo, image_id):
        tile, info = self.vs_native.preprocess_native(photo, image_id, self.pcfg, self.seed)
        vh, vw, c, a = info['valid_h'], info['valid_w'], self.crop, self.align
        top, left = ((vh - c) // 2) // a * a, ((vw - c) // 2) // a * a    # same as the test loader
        crop = np.ascontiguousarray(tile[top:top + c, left:left + c])
        return self.torch.from_numpy(crop).permute(2, 0, 1).float().div(255).unsqueeze(0)

    def __call__(self, photo, image_id=''):
        x = self.preprocess(photo, image_id).to(self.device)
        with self.torch.no_grad():
            out = self.model(x)
        return {'p_ai': float(out['prob'][0]), 'features': out['features'][0].cpu().numpy()}


class M2Detector:
    """M2 (Supriya): frozen CLIP ViT-B/16 image encoder + a small MLP head (models/m2_clip.py).

    The checkpoint holds only the head ({"head", "config", "clip_name", "feature_dim"}); CLIP itself is
    downloaded from Hugging Face. The preprocessing is M2's own, copied from
    preprocessing/build_m2_caches_from_splits.py and preprocessing/datasets.NativeCacheDataset: JPEG
    re-encode at a quality fixed per image (SHA-256 of the image id, 75-95, 4:2:0), centre 256 tile,
    centre 224 crop (aligned to 8), CLIP normalisation, CLIP image features, head, sigmoid.
    """
    name = 'm2_clip'
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, checkpoint, device=None):
        import torch
        from torch import nn
        from transformers import CLIPModel
        self.torch = torch
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        cfg = ck['config']
        self.clip_name = ck.get('clip_name', cfg['model'].get('clip_name', 'openai/clip-vit-base-patch16'))
        self.clip = CLIPModel.from_pretrained(self.clip_name).to(self.device).eval()
        dim = int(ck.get('feature_dim', self.clip.config.projection_dim))
        head_cfg = cfg['model'].get('head', {})
        hidden, dropout = int(head_cfg.get('hidden', 256)), float(head_cfg.get('dropout', 0.2))
        head = nn.Module()
        head.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        head.load_state_dict(ck['head'])                     # same keys as models/m2_clip.MLPHead
        self.head = head.to(self.device).eval()
        p = cfg.get('preprocess', {})
        self.crop, self.align, self.tile = int(p.get('crop_size', 224)), int(p.get('align', 8)), 256
        self.jpeg_low, self.jpeg_high = 75, 95
        self.size = int(self.clip.config.vision_config.image_size)

    def jpeg_quality(self, image_id):
        import hashlib
        digest = hashlib.sha256(str(image_id).encode('utf-8')).digest()
        return self.jpeg_low + (int.from_bytes(digest[:8], 'big') % (self.jpeg_high - self.jpeg_low + 1))

    def preprocess(self, photo, image_id):
        import io
        from PIL import Image
        rgb = photo.convert('RGB')
        width, height = rgb.size
        buffer = io.BytesIO()
        rgb.save(buffer, format='JPEG', quality=self.jpeg_quality(image_id), subsampling=2)
        buffer.seek(0)
        with Image.open(buffer) as decoded_source:
            decoded = np.asarray(decoded_source.convert('RGB'), dtype=np.uint8)
        vh, vw, a, c = min(self.tile, height), min(self.tile, width), self.align, self.crop
        top, left = ((height - vh) // 2) // a * a, ((width - vw) // 2) // a * a
        tile = np.zeros((self.tile, self.tile, 3), dtype=np.uint8)          # as in M2's native cache
        tile[:vh, :vw] = decoded[top:top + vh, left:left + vw]
        # NativeCacheDataset crops the centre of the full 256 tile: its cache JSON keeps the valid size per
        # row, not at the top level, so it never finds it. A photo under 256 px on a side therefore gets a
        # little zero padding in its crop, exactly as M2 saw it in training and evaluation (9 of the 1,300
        # GenImage test photos).
        off = ((self.tile - c) // 2) // a * a
        crop = np.ascontiguousarray(tile[off:off + c, off:off + c])
        return self.torch.from_numpy(crop).permute(2, 0, 1).float().div(255).unsqueeze(0)

    def __call__(self, photo, image_id=''):
        torch = self.torch
        x = self.preprocess(photo, image_id).to(self.device)
        mean = torch.tensor(self.CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(self.CLIP_STD, device=self.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        if x.shape[-2:] != (self.size, self.size):
            x = torch.nn.functional.interpolate(x, size=(self.size, self.size), mode='bicubic', align_corners=False)
        with torch.no_grad():
            out = self.clip.get_image_features(pixel_values=x)
            if hasattr(out, 'pooler_output'):                # transformers 5.x, as in models/m2_clip.py
                out = out.pooler_output
            elif isinstance(out, tuple):
                out = out[0]
            p = torch.sigmoid(self.head.net(out.float()).squeeze(-1))
        return {'p_ai': float(p[0]), 'features': out[0].cpu().numpy()}


def late_fusion(m1_out, m2_out):
    """Late Fusion from README_FUSION.md: the average of the M1 and M2 probabilities."""
    return {'p_ai': (m1_out['p_ai'] + m2_out['p_ai']) / 2, 'features': None}


class DetectorSet:
    """Runs the available detectors on one photo and returns {name: p_ai}.

    detectors: {name: callable}. fusion: optional callable(out_a, out_b) -> {'p_ai'}, applied to
    the two detectors named in fusion_inputs. Missing detectors are simply left out.
    """

    def __init__(self, detectors, fusion=None, fusion_inputs=('m1_resnet', 'm2_clip')):
        self.detectors, self.fusion, self.fusion_inputs = dict(detectors), fusion, fusion_inputs

    @property
    def names(self):
        extra = ['fusion'] if self.fusion and all(n in self.detectors for n in self.fusion_inputs) else []
        return list(self.detectors) + extra

    def __call__(self, photo, image_id=''):
        outs = {n: d(photo, image_id) for n, d in self.detectors.items()}
        scores = {n: o['p_ai'] for n, o in outs.items()}
        if 'fusion' in self.names:
            scores['fusion'] = self.fusion(*(outs[n] for n in self.fusion_inputs))['p_ai']
        return scores
