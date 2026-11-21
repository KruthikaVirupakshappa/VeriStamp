"""Detector wrappers for RQ3 (task 9) and the pipeline (task 10).

Every detector here is a callable: detector(photo: PIL.Image, image_id: str) -> {'p_ai': float,
'features': np.ndarray or None}. image_id must be the same for a photo before and after stamping,
because M1's preprocessing picks its JPEG quality from it.

  DummyDetector  works now. Its score comes from the photo's fine detail, so it reacts a little to
                 the stamp, like a frequency detector would. It means nothing about Real vs AI.
  M1Detector     the real M1 (Kruthika), loaded from the M1 notebook's own modules and best.pt.
                 Its preprocessing is M1's own preprocess_native + centre 224 crop.
                 Confirm the paths and steps with Kruthika before using its numbers.
  M2Detector     placeholder until Supriya's model is ready: fill in _load and __call__.
  late_fusion    average of M1 and M2, the ablation in docs/decisions.md item 11. Use it until the
                 trained fusion model exists, and say so in the report.
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
    """TODO when Supriya's M2 is ready: load the frozen CLIP + MLP head and use her exact
    preprocessing (resize, crop, CLIP normalisation)."""
    name = 'm2_clip'

    def __init__(self, *args, **kwargs):
        raise NotImplementedError('M2 is not ready yet. Ask Supriya for the checkpoint and preprocessing.')

    def __call__(self, photo, image_id=''):
        raise NotImplementedError


def late_fusion(m1_out, m2_out):
    """Average of the two probabilities (late fusion ablation). Replace with the trained fusion model."""
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
