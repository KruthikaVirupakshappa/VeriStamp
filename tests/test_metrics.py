import math

import torch

from evaluation import metrics as M


def test_auroc_known_answer():
    assert M.auroc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == 0.75


def test_decided_metrics():
    labels = [0, 0, 1, 1, 1]
    verdicts = ["Real", "AI", "AI", "Uncertain", "Real"]
    m = M.decided_metrics(labels, verdicts)
    assert m["coverage"] == 0.8
    assert m["accuracy_decided"] == 0.5
    assert m["precision_ai"] == 0.5
    assert math.isclose(m["recall_ai"], 1 / 3)
    assert m["ai_called_real"] == 1


def test_bits_and_psnr():
    a = torch.tensor([[0, 1, 1, 0]])
    assert M.bit_accuracy(a, torch.tensor([[0, 1, 0, 0]])) == 0.75
    x = torch.zeros(1, 3, 4, 4)
    assert math.isclose(M.psnr(x, x + 0.01), 40.0, rel_tol=1e-6)


def test_system_metrics():
    assert M.flip_rate(["Real", "AI"], ["AI", "AI"]) == 0.5
    assert M.ai_stamped_as_real([1, 1, 0], ["Real", "AI", "Real"]) == 1
    assert M.agreement(["Real", None], ["Real", "Real"]) == 0.5
    assert M.invalidation_rate([True, False, False, True]) == 0.5
