from tests.integration import run


def test_pipeline_end_to_end():
    res = run(n=8)
    assert res["none"]["valid"] >= 0.9          # an unedited stamp must decode
    assert res["none"]["bit_acc"] >= 0.99
