import pytest
import numpy as np
from dvq.statistical.JS_divergence import js_divergence


def test_js_identity():
    """
    Identity: JS(seq, seq) should be 0.
    Test that comparing a sequence to itself yields zero divergence.
    """
    seq = "ACGT" * 1000
    assert np.isclose(js_divergence(seq,seq), 0.0)

def test_js_symmetry():
    """got status
    Symmetry: JS(seq_1, seq_2) == JS(seq_2, seq_1)
    Test that JS Divergence is symmetrical.
    """
    seq_1 = "ACGT" * 1000
    seq_2 = "GCTA" * 1000
    assert np.isclose(js_divergence(seq_1, seq_2), js_divergence(seq_2, seq_1))

def test_js_non_negative():
    """
    Non-negativity: JS(seq_1, seq_2) >= 0 for all distributions seq_1, seq_2.
    Test that JS divergence is non-negative.
    """
    seq_1 = "ACGT" * 1000
    seq_2 = "GCTA" * 1000
    assert js_divergence(seq_1, seq_2) >= 0.0
