import pandas as pd
import pytest
from better.benchmark import Benchmark


def test_constructor_with_bad_model_coefficient_type():
    with pytest.raises(ValueError):
        Benchmark('a', 5.0, 5.0, 5.0)


def test_constructor_with_beta_hdd():
    bm = Benchmark('beta_hdd', 4.0, 3.0, 1.0, True)

    assert bm.model_coefficient == 4.0
    assert bm.model_coefficient_type == 'beta_hdd'
    assert bm.sample_median == 3.0
    assert bm.sample_standard_deviation == 1.0
    assert bm.rating == 0
    assert bm.rating_str == 'Typical'


def test_constructor_with_beta_betc():
    bm = Benchmark('beta_betc', 1.0, 3.0, 1.0, True)

    assert bm.model_coefficient == 1.0
    assert bm.model_coefficient_type == 'beta_betc'
    assert bm.sample_median == 3.0
    assert bm.sample_standard_deviation == 1.0
    assert bm.rating == 1
    assert bm.rating_str == 'Poor'
