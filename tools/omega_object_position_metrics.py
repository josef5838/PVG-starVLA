"""Directional responses; zero reference displacement has no direction or gain."""

import numpy as np


def response_metrics(actual, expected):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    actual_norm, expected_norm = float(np.linalg.norm(actual)), float(np.linalg.norm(expected))
    dot = float(np.dot(actual, expected))
    return dict(actual_norm_m=actual_norm, expected_norm_m=expected_norm,
                response_error_m=float(np.linalg.norm(actual - expected)),
                gain=dot / expected_norm ** 2 if expected_norm > 1e-9 else None,
                cosine=dot / (actual_norm * expected_norm) if actual_norm * expected_norm > 1e-12 else None)
