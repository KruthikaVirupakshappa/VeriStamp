import pytest

from watermarking import payload


@pytest.mark.parametrize("verdict", ["Real", "AI", "Uncertain"])
def test_round_trip(verdict):
    bits = payload.encode(verdict, p_ai=0.8, detector="fusion", image_tag=1234)
    assert len(bits) == 32 and set(bits) <= {0, 1}
    out = payload.decode(bits)
    assert out["valid"] and out["verdict"] == verdict
    assert out["p_ai_bucket"] == 12 and out["detector_id"] == 3 and out["image_tag"] == 1234


def test_any_single_flip_is_detected():
    bits = payload.encode("Real", 0.1, "m1_resnet", 77)
    for i in range(32):
        damaged = bits.copy()
        damaged[i] ^= 1
        assert not payload.decode(damaged)["valid"], f"flip at bit {i} not detected"


def test_reserved_verdict_code_is_invalid():
    data = [1, 1] + [0] * 22
    assert not payload.decode(data + payload.crc8(data))["valid"]


def test_bad_inputs():
    with pytest.raises(ValueError):
        payload.encode("Fake", 0.5, "fusion")
    with pytest.raises(ValueError):
        payload.decode([0] * 31)
