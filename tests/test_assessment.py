import numpy as np
import pytest
from better.assessment import OpportunityEngine


@pytest.fixture
def test_sample_benchmark_stats_e():
    return {'beta_base': {'beta_median': 0.352, 'beta_standard_deviation': 0.042105175},
            'beta_cdd': {'beta_median': 0.008635, 'beta_standard_deviation': 0.003189988},
            'beta_betc': {'beta_median': 11.8, 'beta_standard_deviation': 5.128711432},
            'beta_hdd': {'beta_median': 0.00609, 'beta_standard_deviation': 0.00546208},
            'beta_beth': {'beta_median': 13.3, 'beta_standard_deviation': 5.159961489}}


@pytest.fixture
def test_sample_benchmark_stats_f():
    return {'beta_base': {'beta_median': 0.005805,
                          'beta_standard_deviation': 0.008327917},
            'beta_cdd': {'beta_median': 0.0, 'beta_standard_deviation': 0.0},
            'beta_betc': {'beta_median': 0.0, 'beta_standard_deviation': 0.0},
            'beta_hdd': {'beta_median': 0.00698, 'beta_standard_deviation': 0.00901039},
            'beta_beth': {'beta_median': 13.5, 'beta_standard_deviation': 5.965051821}}


@pytest.fixture
def test_fim(test_sample_benchmark_stats_e: dict):
    benchmark_stats = test_sample_benchmark_stats_e.copy()
    test_site_coeffs = {'beta_base': 0, 'beta_cdd': 2,
                        'beta_betc': 3, 'beta_hdd': 0, 'beta_beth': 0}

    for key in test_site_coeffs:
        benchmark_stats[key]['site_coefficient'] = test_site_coeffs[key]

    return OpportunityEngine(benchmark_stats, 'electric')


@pytest.fixture
def test_fim_1(test_sample_benchmark_stats_e: dict):
    benchmark_stats = test_sample_benchmark_stats_e.copy()
    test_site_coeffs = {'beta_base': 0.4124275744214844, 'beta_cdd': 0.01496543169958417,
                        'beta_betc': 22.714713880931324, 'beta_hdd': np.nan, 'beta_beth': np.nan}

    for key in test_site_coeffs:
        benchmark_stats[key]['site_coefficient'] = test_site_coeffs[key]

    return OpportunityEngine(benchmark_stats, 'electric')


@pytest.fixture
def test_fim_2(test_sample_benchmark_stats_f: dict):
    benchmark_stats = test_sample_benchmark_stats_f.copy()
    test_site_coeffs = {'beta_base': 0.003456493309952499, 'beta_cdd': np.nan,
                        'beta_betc': np.nan, 'beta_hdd': 0.00010721179032648839, 'beta_beth': 26.905343586892123}

    for key in test_site_coeffs:
        benchmark_stats[key]['site_coefficient'] = test_site_coeffs[key]

    return OpportunityEngine(benchmark_stats, 'fossil_fuel')


def test_constructor(test_fim: OpportunityEngine, test_fim_2: OpportunityEngine):
    assert test_fim.utility_type == 'electric'
    assert test_fim.base == 0
    assert test_fim.cdd == 2
    assert test_fim.betc == 3
    assert test_fim.hdd == 0
    assert test_fim.beth == 0

    assert test_fim_2.utility_type == 'fossil_fuel'
    expected = {'beta_base': 0.003456493309952499, 'beta_cdd': np.nan,
                'beta_betc': np.nan, 'beta_hdd': 0.00010721179032648839, 'beta_beth': 26.905343586892123}
    for k, v in test_fim_2.benchmark_stats.items():
        assert v['site_coefficient'] == pytest.approx(expected[k], nan_ok=True)


def test_set_targets(test_fim: OpportunityEngine, test_fim_1: OpportunityEngine, test_fim_2: OpportunityEngine):
    # Check for target level 1
    test_fim_1.set_targets(target_level='conservative')

    expected_1 = {'beta_base': 3.94105175e-01, 'beta_cdd': 1.18249880e-02,
                  'beta_betc': 2.27147139e+01, 'beta_hdd': np.nan, 'beta_beth': np.nan}
    for k, v in test_fim_1.benchmark_stats.items():
        assert v['target'] == pytest.approx(expected_1[k], nan_ok=True)

    # Check all values for target level 2
    test_fim.set_targets(target_level='nominal')
    test_fim_2.set_targets(target_level='nominal')

    expected_2 = {'beta_base': 3.52000000e-01, 'beta_cdd': 8.63500000e-03,
                  'beta_betc': 2.27147139e+01, 'beta_hdd': np.nan, 'beta_beth': np.nan}
    for k, v in test_fim.benchmark_stats.items():
        assert v['target'] == pytest.approx(expected_2[k], nan_ok=True)

    expected_3 = {'beta_base': 3.45649331e-03, 'beta_cdd': np.nan,
                  'beta_betc': np.nan, 'beta_hdd': 1.07211790e-04, 'beta_beth': 1.35000000e+01}
    for k, v in test_fim_2.benchmark_stats.items():
        assert v['target'] == pytest.approx(expected_3[k], nan_ok=True)

    # Check for target level 3
    test_fim_1.set_targets(target_level='aggressive')

    expected_4 = {'beta_base': 3.30947412e-01, 'beta_cdd': 7.04000600e-03,
                  'beta_betc': 2.27147139e+01, 'beta_hdd': np.nan, 'beta_beth': np.nan}
    for k, v in test_fim_1.benchmark_stats.items():
        assert v['target'] == pytest.approx(expected_4[k], nan_ok=True)


def test_calculate_recommendations(test_fim: OpportunityEngine):
    test_fim.set_targets(target_level='nominal')
    recommendations = test_fim.calculate_recommendations()

    expected = {'Increase Cooling Setpoints': True,
                'Decrease Heating Setpoints': True,
                'Reduce Equipment Schedules': True,
                'Decrease Ventilation': True,
                'Eliminate Electric Heating': False,
                'Decrease Infiltration': True,
                'Reduce Lighting Load': False,
                'Reduce Plug Loads': False,
                'Add/Fix Economizers': True,
                'Increase Cooling System Efficiency': True,
                'Increase Heating System Efficiency': False,
                'Add Wall/Ceiling Insulation': True,
                'Upgrade Windows': False,
                'Check Fossil Baseload': False}

    assert recommendations == expected


def test_savings_coefficients(test_fim: OpportunityEngine):
    test_fim.set_targets(target_level='nominal')
    test_fim.calculate_recommendations()
    savings_coefficients = test_fim.savings_coefficients()

    expected = {'beta_base': 0, 'beta_cdd': 0.008635,
                'beta_betc': 11.8, 'beta_hdd': 0, 'beta_beth': 0}

    assert {k: v['savings_coefficient']
            for k, v in savings_coefficients.items()} == expected
