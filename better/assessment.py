'''

Energy Efficiency Targeting Tool Copyright (c) 2018, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Intellectual Property Office at  IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights. As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit other to do so. 

'''
from typing import Literal
import pandas as pd
import numpy as np


class OpportunityEngine:
    def __init__(self,
                 benchmark_stats: dict,
                 utility_type: Literal['electric', 'fossil_fuel']):

        self.benchmark_stats = benchmark_stats
        self.utility_type = utility_type

        # Adapt the model coefficients to the OpportunityEngine module
        if self.benchmark_stats['beta_betc']['site_coefficient'] == self.benchmark_stats['beta_beth']['site_coefficient']:
            if self.benchmark_stats['beta_cdd']['site_coefficient'] == 0:
                self.benchmark_stats['beta_betc']['site_coefficient'] = np.nan
                self.benchmark_stats['beta_cdd']['site_coefficient'] = np.nan
            if self.benchmark_stats['beta_hdd']['site_coefficient'] == 0:
                self.benchmark_stats['beta_beth']['site_coefficient'] = np.nan
                self.benchmark_stats['beta_hdd']['site_coefficient'] = np.nan

        self.base = self.benchmark_stats['beta_base']['site_coefficient']
        self.cdd = self.benchmark_stats['beta_cdd']['site_coefficient']
        self.betc = self.benchmark_stats['beta_betc']['site_coefficient']
        self.hdd = self.benchmark_stats['beta_hdd']['site_coefficient']
        self.beth = self.benchmark_stats['beta_beth']['site_coefficient']

    def set_targets(self, target_level: Literal['conservative', 'nominal', 'aggressive']):
        """Set Targets for Opportunity Engine"""

        print('---------------------------------------------------------------')
        print(self.utility_type)
        print('Site Coefficients:')
        [print(k, v['site_coefficient'])
         for k, v in self.benchmark_stats.items()]

        if target_level == 'conservative':
            # Target is 1 standard deviation above the median for each coefficient
            for coefficient_name, coefficient_values in self.benchmark_stats.items():
                if np.isnan(coefficient_values['site_coefficient']):
                    self.benchmark_stats[coefficient_name]['target'] = np.nan
                elif coefficient_name == 'beta_betc':
                    self.benchmark_stats[coefficient_name]['target'] = max(coefficient_values['beta_median'] - coefficient_values['beta_standard_deviation'],
                                                                           coefficient_values['site_coefficient'])
                else:
                    self.benchmark_stats[coefficient_name]['target'] = min(coefficient_values['beta_median'] + coefficient_values['beta_standard_deviation'],
                                                                           coefficient_values['site_coefficient'])

        elif target_level == 'nominal':
            # Target is the median for each coefficient
            for coefficient_name, coefficient_values in self.benchmark_stats.items():
                if np.isnan(coefficient_values['site_coefficient']):
                    self.benchmark_stats[coefficient_name]['target'] = np.nan
                elif coefficient_name == 'beta_betc':
                    self.benchmark_stats[coefficient_name]['target'] = max(coefficient_values['beta_median'],
                                                                           coefficient_values['site_coefficient'])
                else:
                    self.benchmark_stats[coefficient_name]['target'] = min(coefficient_values['beta_median'],
                                                                           coefficient_values['site_coefficient'])

        elif target_level == 'aggressive':
            # Target is half a standard deviation below the median for each coefficient
            for coefficient_name, coefficient_values in self.benchmark_stats.items():
                if np.isnan(coefficient_values['site_coefficient']):
                    self.benchmark_stats[coefficient_name]['target'] = np.nan
                elif coefficient_name == 'beta_betc':
                    self.benchmark_stats[coefficient_name]['target'] = max(coefficient_values['beta_median'] + 0.5 * coefficient_values['beta_standard_deviation'],
                                                                           coefficient_values['site_coefficient'])
                else:
                    self.benchmark_stats[coefficient_name]['target'] = min(coefficient_values['beta_median'] - 0.5 * coefficient_values['beta_standard_deviation'],
                                                                           coefficient_values['site_coefficient'])

        self.base_targ = self.benchmark_stats['beta_base']['target']
        self.cdd_targ = self.benchmark_stats['beta_cdd']['target']
        self.betc_targ = self.benchmark_stats['beta_betc']['target']
        self.hdd_targ = self.benchmark_stats['beta_hdd']['target']
        self.beth_targ = self.benchmark_stats['beta_beth']['target']

        print('Target Coefficients:')
        [print(k, v['target'])
         for k, v in self.benchmark_stats.items()]
        print('---------------------------------------------------------------')

    def calculate_recommendations(self) -> dict:
        # recs = ['Increase Cooling Setpoints', 'Decrease Heating Setpoints',
        #         'Reduce Equipment Schedules', 'Decrease Ventilation',
        #         'Eliminate Electric Heating', 'Decrease Infiltration',
        #         'Reduce Lighting Load', 'Reduce Plug Loads', 'Add/Fix Economizers',
        #         'Increase Cooling System Efficiency',
        #         'Increase Heating System Efficiency', 'Add Wall/Ceiling Insulation',
        #         'Upgrade Windows', 'Check Fossil Baseload']

        self.recommendations = {}

        override_threshold = True
        override_value = 0.001

        setpoint_recommendation = False  # used for scheduling measure

        # region Increase Cooling Setpoint (indicated by low betc)
        recommendation_name = 'Increase Cooling Setpoints'
        threshold = override_value if override_threshold else 0.2

        if (self.betc_targ - self.betc) >= (threshold * self.betc_targ):
            # used for scheduling measure
            self.recommendations[recommendation_name] = True
            setpoint_recommendation = True
        else:
            self.recommendations[recommendation_name] = False

        # endregion

        # region Decrease Heating Setpoint (indicated by high beth)
        recommendation_name = 'Decrease Heating Setpoints'
        threshold = override_value if override_threshold else 0.2

        if (self.beth - self.beth_targ) >= (threshold * self.beth_targ):
            self.recommendations[recommendation_name] = True
            setpoint_recommendation = True
        else:
            self.recommendations[recommendation_name] = False
        # endregion

        # region Tighten Schedules (indicated by high electric baseload)
        """
        9/20/13 - Added logic to also recommend schedules if recommending
        increasing cooling setpoints or decreasing heating setpoints. A
        building's break-even temperature are affected by the average building
        temperatures (i.e., occupied & unoccupied). That means that adjusting
        schedules for switching between an occupied and unoccupied setpoint
        will also change the average building temperatures.
        """
        recommendation_name = 'Reduce Equipment Schedules'
        threshold = override_value if override_threshold else 0.001

        if self.utility_type == 1 and self.base > 0 and (self.base - self.base_targ) >= (threshold * self.base_targ):
            self.recommendations[recommendation_name] = True
        elif setpoint_recommendation:
            self.recommendations[recommendation_name] = True
        else:
            self.recommendations[recommendation_name] = False

        # endregion

        # region Decrease Ventilation (indicated by two of the following three: high cdd, high hdd, high beth)
        recommendation_name = 'Decrease Ventilation'
        count = 0
        threshold = override_value if override_threshold else 0.1

        if self.cdd > 0 and (self.cdd - self.cdd_targ) >= (threshold * self.cdd_targ):
            count += 1
        if self.hdd > 0 and (self.hdd - self.hdd_targ) >= (threshold * self.hdd_targ):
            count += 1

        threshold = override_value if override_threshold else 0.2

        if (self.beth - self.beth_targ) >= (threshold * self.beth_targ):
            count += 1

        self.recommendations[recommendation_name] = count >= 2

        # endregion

        # region Eliminate Any Electric Heating
        """
        Before 9/20/2013, electric heating indicated by electric fuel and HDD>0.
        Added threshold to improve diagnostic and reduce mis-diagnosis of electric
        heating. It appears that both electric and fossil heating has beta_hdd's
        around 0.04 kWh/m2 while electric beta_hdd's without electric heating
        are around 0.004 kWh/m2.
        """
        recommendation_name = 'Eliminate Electric Heating'
        electric_htg_threshold = 0.01

        if self.utility_type == 'electric' and self.hdd > electric_htg_threshold:
            self.recommendations[recommendation_name] = True
        else:
            self.recommendations[recommendation_name] = False

        # endregion

        # region Decrease Infiltration (indicated by two of the following three: high cdd, high hdd, high beth)
        recommendation_name = 'Decrease Infiltration'
        count = 0
        threshold = override_value if override_threshold else 0.1

        if self.cdd > 0 and (self.cdd - self.cdd_targ) >= (threshold * self.cdd_targ):
            count += 1
        if self.hdd > 0 and (self.hdd - self.hdd_targ) >= (threshold * self.hdd_targ):
            count += 1

        threshold = override_value if override_threshold else 0.2

        if (self.beth - self.beth_targ) >= (threshold * self.beth_targ):
            count = count + 1

        self.recommendations[recommendation_name] = count >= 2

        # endregion

        # region Reduce Lighting Load (indicated by high baseload)
        recommendation_name = 'Reduce Lighting Load'
        threshold = override_value if override_threshold else 0.001

        if self.utility_type == 1 and self.base > 0 and (self.base - self.base_targ) >= (threshold * self.base_targ):
            self.recommendations[recommendation_name] = True
        else:
            self.recommendations[recommendation_name] = False

        # endregion

        # region Reduce Plug Load (indicated by high baseload)
        recommendation_name = 'Reduce Plug Loads'
        threshold = override_value if override_threshold else 0.001

        if self.utility_type == 1 and self.base > 0 and (self.base - self.base_targ) >= (threshold * self.base_targ):
            self.recommendations[recommendation_name] = True
        else:
            self.recommendations[recommendation_name] = False

        # endregion

        # region Add/Fix Economizers (indicated by low betc)
        recommendation_name = 'Add/Fix Economizers'
        threshold = override_value if override_threshold else 0.2

        if (self.betc_targ - self.betc) >= (threshold * self.betc_targ):
            self.recommendations[recommendation_name] = True
        else:
            self.recommendations[recommendation_name] = False

        # endregion

        # region Increase Cooling Efficiency (indicated by high cdd)
        recommendation_name = 'Increase Cooling System Efficiency'
        threshold = override_value if override_threshold else 0.1

        if self.cdd > 0 and (self.cdd - self.cdd_targ) >= (threshold * self.cdd_targ):
            self.recommendations[recommendation_name] = True
        else:
            self.recommendations[recommendation_name] = False

        # endregion

        # region Increase Heating Efficiency (indicated by high hdd)
        recommendation_name = 'Increase Heating System Efficiency'
        threshold = override_value if override_threshold else 0.1

        if self.hdd > 0 and (self.hdd - self.hdd_targ) >= (threshold * self.hdd_targ):
            self.recommendations[recommendation_name] = True
        else:
            self.recommendations[recommendation_name] = False

        # endregion

        # region Add Wall/Ceiling Insulation (indicated by two of the following three: high cdd, high hdd, high beth)
        recommendation_name = 'Add Wall/Ceiling Insulation'
        count = 0
        threshold = override_value if override_threshold else 0.1

        if self.cdd > 0 and (self.cdd - self.cdd_targ) >= (threshold * self.cdd_targ):
            count += 1
        if self.hdd > 0 and (self.hdd - self.hdd_targ) >= (threshold * self.hdd_targ):
            count += 1

        threshold = override_value if override_threshold else 0.2

        if (self.beth - self.beth_targ) >= (threshold * self.beth_targ):
            count = count + 1

        self.recommendations[recommendation_name] = count >= 2

        # endregion

        # region Upgrade Windows (indicated by all of the following: high cdd, low betc, high hdd)
        recommendation_name = 'Upgrade Windows'
        count = 0
        threshold = override_value if override_threshold else 0.1

        if self.cdd > 0 and (self.cdd - self.cdd_targ) >= (threshold * self.cdd_targ):
            count += 1
        if self.hdd > 0 and (self.hdd - self.hdd_targ) >= (threshold * self.hdd_targ):
            count += 1

        threshold = override_value if override_threshold else 0.2

        if (self.betc_targ - self.betc) >= (threshold * self.betc_targ):
            count += 1

        self.recommendations[recommendation_name] = (count == 3)

        # endregion

        # region Check Excessive Fossil Fuel Baseload
        recommendation_name = 'Check Fossil Baseload'
        threshold = override_value if override_threshold else 0.001

        if self.utility_type == 2 and self.base > 0 and (self.base - self.base_targ) >= (threshold * self.base_targ):
            self.recommendations[recommendation_name] = True
        else:
            self.recommendations[recommendation_name] = False

        # endregion

        return self.recommendations

    def savings_coefficients(self) -> dict:

        for coefficient_name, coefficient_values in self.benchmark_stats.items():
            site_coefficient = coefficient_values['site_coefficient']
            target = coefficient_values['target']

            if (site_coefficient > target):
                if (coefficient_name == 'beta_betc'):
                    coefficient_values['savings_coefficient'] = site_coefficient
                else:
                    coefficient_values['savings_coefficient'] = target
            else:
                if (coefficient_name == 'beta_betc'):
                    coefficient_values['savings_coefficient'] = target
                else:
                    coefficient_values['savings_coefficient'] = site_coefficient

        return self.benchmark_stats
