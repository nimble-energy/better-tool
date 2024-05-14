import numpy as np
import numpy.typing as npt
import pytest
from better.model import InverseModel


@pytest.fixture
def test_temperature():
    return np.array([68.12575107, 70.38140704, 71.49038076, 75.91127527, 79.23819562,
                     80.94022825, 83.36044143, 82.32376491, 81.7343778, 79.23531421,
                     74.43723106, 68.94813234])


@pytest.fixture
def test_eui_electric():
    return np.array([0.41082736, 0.41278433, 0.42939531, 0.42957719, 0.4665275,
                     0.496334, 0.48262927, 0.5129985, 0.47703349, 0.45208535,
                     0.43193808, 0.396703])


@pytest.fixture
def test_eui_fossil_fuel():
    return np.array([0.00418866, 0.00408925, 0.00408699, 0.0037231, 0.00357866,
                     0.00344965, 0.00323299, 0.00343633, 0.00347068, 0.00365999,
                     0.00391241, 0.00396499])


def test_constructor_with_different_length_arrays(test_temperature: npt.ArrayLike, test_eui_electric: npt.ArrayLike):
    with pytest.raises(Exception):
        InverseModel(temperature=test_temperature[2:],  # type: ignore
                     eui=test_eui_electric)


def test_constructor(test_temperature: npt.ArrayLike, test_eui_electric: npt.ArrayLike):
    model = InverseModel(temperature=test_temperature,
                         eui=test_eui_electric)

    assert not model.has_fit
    assert model.hcp_bound_percentile == 45
    assert model.ccp_bound_percentile == 55
    assert model.hcp == np.percentile(test_temperature, 45)
    assert model.ccp == np.percentile(test_temperature, 55)


def test_piecewise_linear(test_temperature):
    expected = [49.37124465, 38.0929648, 32.5480962, 10.44362365, 10., 10.,
                21.80220715, 16.61882455, 13.671889, 10., 17.8138447, 45.2593383]
    result = InverseModel.piecewise_linear(test_temperature, 76, 81, 10, -5, 5)

    assert expected == pytest.approx(list(result))


def test_fit_model_electric(test_temperature: npt.ArrayLike, test_eui_electric: npt.ArrayLike):
    model = InverseModel(temperature=test_temperature,
                         eui=test_eui_electric)

    has_fit = model.fit_model()

    assert has_fit
    assert model.base == pytest.approx(0.4124275744214844)
    assert model.ccp == pytest.approx(72.88651279550146)
    assert model.hcp == pytest.approx(72.88651279550146)
    assert model.hsl == 0
    assert model.csl == pytest.approx(0.008314150916399145)
    assert model.model_type_str == '3P Cooling'
    assert model.cp_txt == '(72.9, 0.4)'


def test_fit_model_fossil_fuel(test_temperature: npt.ArrayLike, test_eui_fossil_fuel: npt.ArrayLike):
    model = InverseModel(temperature=test_temperature,
                         eui=test_eui_fossil_fuel)

    has_fit = model.fit_model()

    assert has_fit
    assert model.base == pytest.approx(0.0034564933318507354)
    assert model.ccp == pytest.approx(80.4296184558961)
    assert model.hcp == pytest.approx(80.4296184558961)
    assert model.hsl == pytest.approx(-5.956210102125221e-05)
    assert model.csl == 0
    assert model.model_type_str == '3P Heating'
    assert model.cp_txt == '(80.4, 0.0)'


def test_4p_model():
    temperature = np.array([16.49774436, 19.17597293, 20.54890511, 22.68663594, 24.98291925,
                            27.78870968, 29.52193646, 28.97226754, 27.16207455, 23.29365079,
                            20.12974684, 13.40963855])  # Celsius
    eui = np.array([0.43714513, 0.43216608, 0.42055272, 0.40125029, 0.44447121,
                    0.49665507, 0.49037908, 0.51118835, 0.45466266, 0.42066609,
                    0.40706217, 0.42708896])

    model = InverseModel(temperature=temperature,
                         eui=eui)

    has_fit = model.fit_model()

    assert has_fit
    assert model.base == pytest.approx(0.40628945105882824)
    assert model.ccp == pytest.approx(23.09257295245528)
    assert model.hcp == pytest.approx(23.09257295245528)
    assert model.hsl == pytest.approx(-0.002143103535229683)
    assert model.csl == pytest.approx(0.014274532918350118)
    assert model.model_type_str == '4P'
    assert model.cp_txt == '(23.1, 0.4)'


def test_5p_model():
    temperature = np.array([10.349928430846305,
                            11.973839223839226,
                            12.883942949160346,
                            18.48328416912488,
                            20.052400165528656,
                            23.64396135265701,
                            25.758709016393443,
                            24.970244420828905,
                            21.39081375282092,
                            18.503562680917394,
                            12.435875316310097,
                            8.105382716049382])

    eui = np.array([0.4365708589244377, 0.4134988047159823, 0.3927128726414183, 0.3742807152082102,
                    0.3959484654258048, 0.4403108014723988, 0.43429237715898233, 0.46001543271825485,
                    0.4185359650470618, 0.3747953826734285, 0.38644086357394014, 0.4122658144475816])

    model = InverseModel(temperature=temperature,
                         eui=eui)

    model.fit_model()

    assert model.has_fit
    assert model.base == pytest.approx(0.3748200839395793)
    assert model.ccp == pytest.approx(18.070284144642244)
    assert model.hcp == pytest.approx(17.11723536690772)
    assert model.hsl == pytest.approx(-0.005610068926068599)
    assert model.csl == pytest.approx(0.010430568680742096)
    assert model.model_type_str == '5P'
    assert model.cp_txt == ['(17.1, 0.4)', '(18.1, 0.4)']
    assert model.r2 == pytest.approx(0.7760372322434232)


def test_rmse(test_temperature: npt.ArrayLike, test_eui_electric: npt.ArrayLike):
    model = InverseModel(temperature=test_temperature,
                         eui=test_eui_electric)

    model.fit_model()

    assert model.rmse() == pytest.approx(0.01278258303587747)
