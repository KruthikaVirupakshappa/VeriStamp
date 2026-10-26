import pytest

from models.thresholds import to_verdict


def test_bands():
    assert to_verdict(0.10, 0.35, 0.65) == "Real"
    assert to_verdict(0.50, 0.35, 0.65) == "Uncertain"
    assert to_verdict(0.35, 0.35, 0.65) == "Uncertain"   # boundaries are Uncertain
    assert to_verdict(0.90, 0.35, 0.65) == "AI"


def test_bad_thresholds():
    with pytest.raises(ValueError):
        to_verdict(0.5, 0.7, 0.3)
