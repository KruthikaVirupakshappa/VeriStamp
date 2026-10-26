import torch
from PIL import Image

from common.config import load_config
from preprocessing.datasets import ManifestDataset
from preprocessing.frequency import m1_input
from preprocessing.build_splits import build
from preprocessing.transforms import load_rgb, preprocess

CFG = load_config("configs/base.yaml")


def _img(color, size=(300, 260)):
    return Image.new("RGB", size, color)


def test_output_shape_and_range():
    x = preprocess(_img((120, 30, 200)), "a", CFG)
    assert x.shape == (3, 224, 224)
    assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0


def test_deterministic_per_image_id():
    img = _img((10, 200, 90))
    assert torch.equal(preprocess(img, "same-id", CFG), preprocess(img, "same-id", CFG))


def test_png_and_jpeg_sources_end_up_identical_format(tmp_path):
    """Bias control: a PNG (typical AI file) and a JPEG (typical real file) with the same
    pixels must produce tensors of the same shape, through the same JPEG re-encode."""
    img = _img((200, 120, 40))
    img.save(tmp_path / "a.png")
    img.save(tmp_path / "a.jpg", quality=100)
    xa = preprocess(load_rgb(tmp_path / "a.png"), "x", CFG)
    xb = preprocess(load_rgb(tmp_path / "a.jpg"), "x", CFG)
    assert xa.shape == xb.shape
    assert (xa - xb).abs().mean() < 0.02


def test_m1_input_channels():
    assert m1_input(torch.rand(2, 3, 224, 224)).shape == (2, 4, 224, 224)


def test_splits_and_folds_have_no_leakage(tmp_path):
    cfg = {**CFG, "data": {**CFG["data"], "per_class": 10}}
    for gen in cfg["data"]["generators"]:
        for sub, color in (("nature", (90, 90, 90)), ("ai", (200, 50, 50))):
            d = tmp_path / gen / sub
            d.mkdir(parents=True)
            for i in range(12):
                _img(color).save(d / f"{i}.png")
    df = build(cfg, tmp_path)
    assert len(df) == 3 * 2 * 10
    csv = tmp_path / "m.csv"
    df.to_csv(csv, index=False)
    for fold, held_out in enumerate(cfg["data"]["generators"]):
        train = ManifestDataset(str(csv), "train", cfg, fold).rows
        val = ManifestDataset(str(csv), "val", cfg, fold).rows
        test = ManifestDataset(str(csv), "test", cfg, fold).rows
        assert held_out not in set(train.generator) | set(val.generator)
        assert set(test.generator) == {held_out}
        assert set(train.image_id).isdisjoint(test.image_id)
        assert set(test.label) == {0, 1}
