from cv_starter.model import build_mobilenetv2


def test_model_builds():
    m = build_mobilenetv2(224, 8)
    assert m.output_shape[-1] == 8
