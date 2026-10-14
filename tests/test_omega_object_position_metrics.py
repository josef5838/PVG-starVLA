"""Distinguish correct, opposing, perpendicular and undefined TCP responses."""

from tools.omega_object_position_metrics import response_metrics


def test_response_direction_and_zero_reference():
    same = response_metrics([0.01, 0, 0], [0.02, 0, 0])
    assert abs(same["gain"] - 0.5) < 1e-12 and same["cosine"] == 1
    opposite = response_metrics([-0.02, 0, 0], [0.02, 0, 0])
    assert opposite["gain"] == -1 and opposite["cosine"] == -1
    perpendicular = response_metrics([0, 0.02, 0], [0.02, 0, 0])
    assert perpendicular["gain"] == 0 and perpendicular["cosine"] == 0
    stationary = response_metrics([0, 0, 0], [0.02, 0, 0])
    assert stationary["gain"] == 0 and stationary["cosine"] is None
    no_reference = response_metrics([0.02, 0, 0], [0, 0, 0])
    assert no_reference["gain"] is None and no_reference["cosine"] is None
