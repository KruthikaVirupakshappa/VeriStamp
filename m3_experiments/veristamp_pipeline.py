"""The decode, detect and stamp pipeline from the proposal (task 10). The demo can import this.

    pipe = VeriStampPipeline(mw, m3, detector_set, decide_with='fusion', t_low=0.35, t_high=0.65)
    out = pipe.run(photo, image_id='photo_001')
    out['verdict'], out['source'], out['photo']      # source is 'stamp' or 'detectors'

1. Read the stamp with extract. If it is valid, return its verdict and skip the detectors.
2. Otherwise run the detectors (M1, M2 and the fusion model) and apply the Uncertain thresholds.
3. Stamp the final verdict with embed and return the stamped photo.

A valid stamp only means "our detector said X when this photo was stamped". The stamp is not
signed, so it is not proof that a photo is authentic. Say "valid stamp with stored verdict X" in
the demo, never "verified".
"""
from __future__ import annotations

import time
import zlib

DETECTOR_IDS = {'stub': 0, 'dummy': 0, 'm1_resnet': 1, 'm2_clip': 2, 'fusion': 3}   # = watermarking/payload.py


def to_verdict(p_ai, t_low, t_high):
    """Same rule as models/thresholds.py: below t_low Real, above t_high AI, otherwise Uncertain."""
    if p_ai < t_low:
        return 'Real'
    if p_ai > t_high:
        return 'AI'
    return 'Uncertain'


def image_tag_for(image_id):
    """Same rule as watermarking/payload.py: 14-bit tag from the image id."""
    return zlib.crc32(str(image_id).encode()) & 0x3FFF


class VeriStampPipeline:
    def __init__(self, mw, m3, detector_set, decide_with='fusion', t_low=0.35, t_high=0.65):
        """mw: the m3_watermark module. m3: from mw.load_m3. detector_set: detectors.DetectorSet.
        decide_with: which score makes the verdict, 'fusion' normally, or a single detector as a
        fallback while fusion is not ready."""
        self.mw, self.m3, self.detectors = mw, m3, detector_set
        self.decide_with, self.t_low, self.t_high = decide_with, t_low, t_high

    def run(self, photo, image_id=''):
        start = time.perf_counter()
        photo = photo.convert('RGB')
        stamp = self.mw.extract(self.m3, photo)
        if stamp['valid']:
            return {'source': 'stamp', 'verdict': stamp['verdict'], 'p_ai': stamp['p_ai'],
                    'detector_id': stamp['detector_id'], 'image_tag': stamp['image_tag'],
                    'scores': None, 'photo': photo, 'seconds': time.perf_counter() - start}
        scores = self.detectors(photo, image_id)
        p_ai = scores[self.decide_with]
        verdict = to_verdict(p_ai, self.t_low, self.t_high)
        tag = image_tag_for(image_id)
        stamped = self.mw.embed(self.m3, photo, verdict, p_ai,
                                detector_id=DETECTOR_IDS.get(self.decide_with, 0), image_tag=tag)
        return {'source': 'detectors', 'verdict': verdict, 'p_ai': p_ai,
                'detector_id': DETECTOR_IDS.get(self.decide_with, 0), 'image_tag': tag,
                'scores': scores, 'photo': stamped, 'seconds': time.perf_counter() - start}
