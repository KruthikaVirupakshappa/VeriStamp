# Interfaces between workstreams

Agree on these once. After that, each workstream can build against a stub until the real
implementation lands. Change them only by PR, with all four people agreeing.

```python
# preprocessing/datasets.py                           (Workstream A)
ManifestDataset(csv_path, split, cfg, fold=None)[i] -> (x01: FloatTensor[3,224,224], label: LongTensor[], meta: dict)
#   fold=k: generator k removed from train/val; test contains only generator k

# models/base.py                                      (Workstream B implements, D consumes)
detector.name                      # "m1_resnet" | "m2_clip" | "fusion" | "stub"
detector.predict(x01: [B,3,H,W] in 0..1) -> {"p_ai": FloatTensor[B], "features": FloatTensor[B,D]}

# models/thresholds.py                                (B fits, D evaluates)
to_verdict(p_ai, t_low, t_high) -> "Real" | "AI" | "Uncertain"
load_thresholds("checkpoints/thresholds_<model>.json") -> (t_low, t_high)

# watermarking/payload.py                             (C + D)
encode(verdict, p_ai, detector_name, image_tag) -> list[int]  # 32 bits
decode(bits) -> {"valid", "verdict", "p_ai_bucket", "detector_id", "image_tag"}

# watermarking/base.py                                (Workstream C implements, D consumes)
wm.embed(x01: [B,3,H,W], bits: LongTensor[B,32]) -> x01_stamped (same shape)
wm.extract(x01: [B,3,H,W]) -> logits: FloatTensor[B,32]       # bit = logit > 0

# pipeline.py                                         (Workstream D)
VeriStamp(detector, wm, t_low, t_high)
  .stamp(x01, image_ids) -> (x01_stamped, records)
  .check(x01) -> [{"stamp_valid", "stored_verdict", "fresh_verdict", "p_ai", "reported_verdict", "agree", "bits"}]
```

**Stubs available now:** `models.base.StubDetector` and `watermarking.base.SpreadSpectrumWatermarker`.
