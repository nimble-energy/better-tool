'''

Energy Efficiency Targeting Tool Copyright (c) 2018, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Intellectual Property Office at  IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights. As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit other to do so. 

'''
from typing import Literal
import numpy as np
import numpy.typing as npt
import pandas as pd
# import scipy.stats as st
# import copy


class Benchmark:

    def __init__(self,
                 model_coefficient_type: Literal['beta_hdd', 'beta_beth', 'beta_base', 'beta_betc', 'beta_cdd'],
                 model_coefficient: float,
                 sample_median: float,
                 sample_standard_deviation: float,
                 model_coefficient_valid: bool = False):

        # Confirm that model_coefficient_tye is valid
        valid_model_coefficient_types = [
            'beta_hdd', 'beta_beth', 'beta_base', 'beta_betc', 'beta_cdd']
        if (model_coefficient_type not in valid_model_coefficient_types):
            raise ValueError(
                f"Model coefficient type must be one of {valid_model_coefficient_types}")

        if model_coefficient_valid:
            try:
                # Benchmark (-1: good; 0: typical; 1: poor)
                if (model_coefficient < sample_median - sample_standard_deviation):
                    self.rating = -1
                elif (sample_median - sample_standard_deviation <= model_coefficient <= sample_median + sample_standard_deviation):
                    self.rating = 0
                elif (model_coefficient > sample_median + sample_standard_deviation):
                    self.rating = 1
                if (model_coefficient_type == 'beta_betc'):
                    self.rating *= -1
                if (sample_standard_deviation == 0):
                    self.rating = 0
                # Assign a string
                if (self.rating == -1):
                    self.rating_str = 'Good'
                if (self.rating == 1):
                    self.rating_str = 'Poor'
                if (self.rating == 0):
                    self.rating_str = 'Typical'

            except:
                raise RuntimeError(
                    "Benchmarking is not successful; benchmark distribution not provided.")

        self.model_coefficient_type = model_coefficient_type
        self.model_coefficient = model_coefficient
        self.sample_median = sample_median
        self.sample_standard_deviation = sample_standard_deviation
        self.model_coefficient_valid = model_coefficient_valid

    @staticmethod
    def generate_benchmark_stats(model_coefficient_type: Literal['beta_hdd', 'beta_beth', 'beta_base', 'beta_betc', 'beta_cdd'],
                                 sample_coefficients: npt.ArrayLike):
        # replace NaNs with 0's for heating slope (beta_hdd) only
        if (model_coefficient_type == "beta_hdd"):
            sample_coefficients = np.nan_to_num(sample_coefficients)
        sample_median = np.nanmedian(sample_coefficients)  # type: ignore
        sample_MAD = Benchmark.median_absolute_deviation(sample_coefficients)
        # Coefficient for normally distributed data
        sample_standard_deviation = 1.4826 * sample_MAD
        return sample_median, sample_standard_deviation

    @staticmethod
    def median_absolute_deviation(v_data: npt.ArrayLike):
        return np.nanmedian(abs(v_data - np.nanmedian(v_data)))  # type: ignore

    # @staticmethod
    # def standardize_target(benchmark):
    #     # Standardize the distribution before plotting
    #     if (benchmark.sample_STD != 0):
    #         standard_x = (benchmark.sample_median -
    #                       benchmark.model_coefficient) / benchmark.sample_STD
    #     else:
    #         standard_x = benchmark.sample_median
    #     if (benchmark.model_coefficient_type == 'beta_betc'):
    #         standard_x *= -1
    #     # When the target building is extremely high or low in the distribution
    #     if (standard_x < -3.45):
    #         standard_x = -3.45
    #     if (standard_x > 3.45):
    #         standard_x = 3.45
    #     return (standard_x)

    # @staticmethod
    # def generate_benchmark_bar_html(benchmark):
    #     coeff_str_dict = {'beta_hdd': 'Heating Sensitivity',
    #                       'beta_beth': 'Heating Change-point',
    #                       'beta_base': 'Baseload',
    #                       'beta_betc': 'Cooling Change-point',
    #                       'beta_cdd': 'Cooling Sensitivity'}
    #     if (hasattr(benchmark, 'model_coefficient_type') and benchmark.valid):
    #         coeff_type = coeff_str_dict[benchmark.model_coefficient_type]
    #         standard_z_score = Benchmark.standardize_target(benchmark)
    #         percent = int(round(st.norm.cdf(standard_z_score) * 100, 0))
    #         bench_bar_html = ''
    #         bench_bar_html += '<div class="w3-row">'
    #         bench_bar_html += '  <div class="w3-third">'
    #         bench_bar_html += '    <p style="margin:0px"><b>' + \
    #             coeff_type + '</b> <br/> (' + benchmark.rating_str + ')</p>'
    #         bench_bar_html += '  </div>'
    #         bench_bar_html += '  <div class="w3-twothird">'
    #         if percent < 85:
    #             bench_bar_html += '    <div class="benchmark_bar w3-round-xlarge"><div class="vertical_line"style="left:' + str(
    #                 percent) + '%;"><div class="w3-container w3-padding w3-center">' + str(percent) + '%</div></div></div>'
    #         else:
    #             bench_bar_html += '    <div class="benchmark_bar w3-round-xlarge"><div class="vertical_line"style="left:' + str(
    #                 percent) + '%;"><div class="w3-container w3-padding w3-right">' + str(percent) + '%</div></div></div>'
    #         bench_bar_html += '  </div>'
    #         bench_bar_html += '</div>'
    #     else:
    #         bench_bar_html = ''
    #     return bench_bar_html
