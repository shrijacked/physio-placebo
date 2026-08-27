"""Constants and Templates for Sensor Data Captioning in SensorLM."""

low_level_templates = [
    (
        f'{{name}} mean, max, min, std are {{mean_val:.1f}}, {{max_val:.1f}},'
        f' {{min_val:.1f}}, {{std_val:.1f}}.'
    ),
    (
        f'{{name}} exhibits a mean of {{mean_val:.1f}}, with peak and minimal'
        f' values reaching {{max_val:.1f}} and {{min_val:.1f}}, and a standard'
        f' deviation of {{std_val:.1f}}.'
    ),
    (
        f'{{name}} shows a mean of {{mean_val:.1f}}, with range {{max_val:.1f}}'
        f' to {{min_val:.1f}} and a standard deviation of {{std_val:.1f}}.'
    ),
    (
        f'{{name}} metrics include a mean of {{mean_val:.1f}}, a maximum of'
        f' {{max_val:.1f}}, a minimum of {{min_val:.1f}}, and a standard'
        f' deviation of {{std_val:.1f}}.'
    ),
    (
        f'{{name}} has a mean of {{mean_val:.1f}}, a maximum of'
        f' {{max_val:.1f}}, a minimum of {{min_val:.1f}}, and a standard'
        f' deviation of {{std_val:.1f}}.'
    ),
    (
        f'{{name}} displays a mean of {{mean_val:.1f}}, a maximum of'
        f' {{max_val:.1f}}, a minimum of {{min_val:.1f}}, and a standard'
        f' deviation of {{std_val:.1f}}.'
    ),
    (
        f'{{name}} presents a mean of {{mean_val:.1f}}, a maximum of'
        f' {{max_val:.1f}}, a minimum of {{min_val:.1f}}, and a standard'
        f' deviation of {{std_val:.1f}}.'
    ),
    (
        f'{{name}} reports a mean of {{mean_val:.1f}}, a maximum of'
        f' {{max_val:.1f}}, a minimum of {{min_val:.1f}}, and a standard'
        f' deviation of {{std_val:.1f}}.'
    ),
    (
        f'{{name}} average {{mean_val:.1f}}, reaching a maximum of'
        f' {{max_val:.1f}} and a minimum of {{min_val:.1f}}, with a standard'
        f' deviation of {{std_val:.1f}}.'
    ),
    (
        f'{{name}} features a mean of {{mean_val:.1f}}, a maximum of'
        f' {{max_val:.1f}}, a minimum of {{min_val:.1f}}, and a standard'
        f' deviation of {{std_val:.1f}}.'
    ),
    (
        f'The average {{name}} value is {{mean_val:.1f}}, with extremes at'
        f' {{max_val:.1f}} (max) and {{min_val:.1f}} (min), and a std of'
        f' {{std_val:.1f}}.'
    ),
    (
        f'Statistical overview of {{name}}: mean = {{mean_val:.1f}}, max ='
        f' {{max_val:.1f}}, min = {{min_val:.1f}}, standard deviation ='
        f' {{std_val:.1f}}.'
    ),
    (
        f'{{name}} data summary: average = {{mean_val:.1f}}, highest point ='
        f' {{max_val:.1f}}, lowest point = {{min_val:.1f}}, standard deviation'
        f' = {{std_val:.1f}}.'
    ),
    (
        f'The {{name}} readings show a central value of {{mean_val:.1f}}, a'
        f' standard deviation of {{std_val:.1f}}, and a value range of'
        f' {{min_val:.1f}}–{{max_val:.1f}}.'
    ),
    (
        f'Observed statistics for {{name}}: mean of {{mean_val:.1f}}, standard'
        f' deviation of {{std_val:.1f}}, with values ranging from'
        f' {{min_val:.1f}} to {{max_val:.1f}}.'
    ),
    (
        f'The {{name}} sensor data has a mean of {{mean_val:.1f}}, a standard'
        f' deviation of {{std_val:.1f}}, and its values fluctuate between'
        f' {{min_val:.1f}} and {{max_val:.1f}}.'
    ),
    (
        f'For the {{name}} measurements, the mean is {{mean_val:.1f}}, the'
        f' standard deviation is {{std_val:.1f}}, and the data lies between'
        f' {{min_val:.1f}} and {{max_val:.1f}}.'
    ),
    (
        f'A summary of {{name}} data reveals a mean of {{mean_val:.1f}}, a'
        f' standard deviation of {{std_val:.1f}}, a minimum of {{min_val:.1f}},'
        f' and a maximum of {{max_val:.1f}}.'
    ),
    (
        f'The {{name}} values are characterized by a mean of {{mean_val:.1f}},'
        f' a standard deviation of {{std_val:.1f}}, and a range from'
        f' {{min_val:.1f}} to {{max_val:.1f}}.'
    ),
    (
        f'The {{name}} data exhibits a mean of {{mean_val:.1f}}, a standard'
        f' deviation of {{std_val:.1f}}, and its extreme values are'
        f' {{min_val:.1f}} and {{max_val:.1f}}.'
    ),
]

activity_templates = [
    f'{{activity}} during minutes {{start_minute}} to {{end_minute}}. ',
    f'{{activity}} between minutes {{start_minute}} and {{end_minute}}.',
    (
        f'{{activity}} was detected between minutes {{start_minute}} and'
        f' {{end_minute}}.'
    ),
    (
        f'A {{activity}} event was detected between minutes {{start_minute}}'
        f' and {{end_minute}}.'
    ),
    (
        f'A {{activity}} period between minutes {{start_minute}} and'
        f' {{end_minute}}.'
    ),
    f'{{activity}} occurred from minute {{start_minute}} to {{end_minute}}.',
    (
        f'Observed {{activity}} spanning minutes {{start_minute}} to'
        f' {{end_minute}}.'
    ),
    (
        f'{{activity}} was recorded between minute {{start_minute}} and'
        f' {{end_minute}}.'
    ),
    (
        f'Period of {{activity}} noted from minute {{start_minute}} until'
        f' {{end_minute}}.'
    ),
    (
        f'{{activity}} took place during the minutes {{start_minute}} through'
        f' {{end_minute}}.'
    ),
    (
        f'Identified {{activity}} across the timeframe of minute'
        f' {{start_minute}} to {{end_minute}}.'
    ),
    (
        f'{{activity}} is indicated between the {{start_minute}} and'
        f' {{end_minute}} minute marks.'
    ),
    (
        f'An instance of {{activity}} was identified from minute'
        f' {{start_minute}} to {{end_minute}}.'
    ),
    (
        f'{{activity}} state observed from minute {{start_minute}} up to'
        f' {{end_minute}}.'
    ),
    (
        f'User engaged in {{activity}} between minute {{start_minute}} and'
        f' {{end_minute}}.'
    ),
    (
        f'A continuous {{activity}} phase from minute {{start_minute}} to'
        f' {{end_minute}}.'
    ),
    (
        f'{{activity}} episode occurred between minute {{start_minute}} and'
        f' {{end_minute}}.'
    ),
    (
        f'From minute {{start_minute}} to {{end_minute}}, the user had a period'
        f' of {{activity}}.'
    ),
    (
        f'Detection of {{activity}} activity between minute {{start_minute}}'
        f' and {{end_minute}}.'
    ),
    (
        f'{{activity}} recorded within the {{start_minute}}-{{end_minute}}'
        f' minute range.'
    ),
]


trend_templates = [
    (
        f'From minute {{start}} to {{end}}, {{sensor_name}} exhibits an'
        f' {{trend_type}} trend.'
    ),
    (
        f'{{sensor_name}} is observed to be {{trend_type}} between minute'
        f' {{start}} and {{end}}.'
    ),
    (
        f'During minute {{start}} to {{end}}, there is a {{trend_type}} trend'
        f' in {{sensor_name}}.'
    ),
    (
        f'Data from {{sensor_name}} indicates a {{trend_type}} trend spanning'
        f' minute {{start}} to {{end}}.'
    ),
    (
        f'{{sensor_name}} reveals an {{trend_type}} trend during the interval'
        f' of minute {{start}} to {{end}}.'
    ),
    (
        f'{{sensor_name}} is {{trend_type}} throughout the period from minute'
        f' {{start}} to minute {{end}}.'
    ),
    (
        f'Over minute {{start}} to {{end}}, {{sensor_name}} displays a'
        f' {{trend_type}} trend.'
    ),
    (
        f'An {{trend_type}} trend in {{sensor_name}} data recorded between'
        f' minute {{start}} and {{end}}.'
    ),
    (
        f'{{sensor_name}} undergoes an {{trend_type}} trend in the time window'
        f' of minute {{start}} to {{end}}.'
    ),
    f'{{sensor_name}} is {{trend_type}} between minutes {{start}}-{{end}}.',
    (
        f'During the period of minute {{start}} to {{end}}, {{sensor_name}} is'
        f' {{trend_type}}.'
    ),
    (
        f'Observing a {{trend_type}} trend in {{sensor_name}} across minutes'
        f' {{start}} to {{end}}.'
    ),
    (
        f'{{trend_type}} trend in {{sensor_name}} observed between minutes'
        f' {{start}} and {{end}}.'
    ),
    (
        f'{{sensor_name}} exhibits {{trend_type}} trend during minute'
        f' {{start}}-{{end}} interval.'
    ),
    (
        f'The {{sensor_name}} trend from minute {{start}} to {{end}} is'
        f' {{trend_type}}.'
    ),
]


anomaly_templates = [
    f'A {{anomaly}} is detected for {{sensor_name}} at minute {{time}}.',
    f'At minute {{time}}, {{sensor_name}} shows a {{anomaly}}.',
    f'We observe a {{anomaly}} in {{sensor_name}} data at minute {{time}}.',
    f'Minute {{time}} shows a {{anomaly}} for the {{sensor_name}}.',
    f'The {{sensor_name}} experienced a {{anomaly}} at minute {{time}}.',
    f'A {{anomaly}} is recorded for {{sensor_name}} at minute {{time}}.',
    f'A {{anomaly}} occurs in the {{sensor_name}} readings at minute {{time}}.',
    (
        f'Data indicates a {{anomaly}} for {{sensor_name}} at the'
        f' {{time}}-minute mark.'
    ),
    f'A {{anomaly}} observed in {{sensor_name}} at minute {{time}}.',
    f'{{sensor_name}} displays a {{anomaly}} at minute {{time}}.',
    f'Irregular {{anomaly}} in {{sensor_name}} readings at minute {{time}}.',
    f'At minute {{time}}, an {{anomaly}} is present in {{sensor_name}}.',
    f'Unusual {{anomaly}} in {{sensor_name}} data at minute {{time}}.',
    f'Noteworthy {{anomaly}} for {{sensor_name}} at minute {{time}}.',
    f'{{anomaly}} event recorded for {{sensor_name}} at minute {{time}}.',
]

mood_templates_with_time = [
    f'The person logged their mood as {{mood}} at minute {{time}}.',
    f'A mood of {{mood}} was logged at minute {{time}}.',
    f'Mood appears to be {{mood}} at minute {{time}}.',
    f"At minute {{time}}, the person's mood is {{mood}}.",
    f'Feeling {{mood}} was reported by the individual at minute {{time}}.',
    f'The individual feels {{mood}} at minute {{time}}.',
    f'Reported {{mood}} mood at minute {{time}} for the individual.',
    f'The person registered a mood of {{mood}} at minute {{time}}.',
    f"At minute {{time}}, the individual's feeling is {{mood}}.",
    f'At minute {{time}}, the person logs their mood as {{mood}}.',
]


NORMALIZATION_PARAMETERS = {
    'HR': [75.9586, 16.1887],
    'eda_level_real': [4.1767, 5.5893],
    'leads_contact_counts': [226.4864, 67.3312],
    'steps': [5.1679, 18.8926],
    'jerk_auto': [203.4672, 30.0563],
    'step_count': [10.972891950490821, 16.380615326908973],
    'log_energy': [53.0804, 49.6526],
    'covariance': [43.4077, 13.9529],
    'log_energy_ratio': [44.8483, 22.9746],
    'zero_crossing_std': [155.1863, 28.2378],
    'zero_crossing_avg': [51.0043, 37.4756],
    'axis_mean': [123.1659, 21.4710],
    'altim_std': [0.0042, 0.0597],
    'kurtosis': [105.5954, 66.8495],
    'sleep_coefficient': [7.2623, 5.3946],
    'wrist_temperatures': [31.6745, 2.5789],
    'hrv_shannon_entropy_rr': [3.2953304951132885, 0.464777365409023],
    'hrv_shannon_entropy_rrd': [2.9810634995051184, 0.48817021363471297],
    'hrv_percentage_of_nn_30': [0.35201734277287905, 0.1902607735053669],
    'ceda_magnitude_real_micro_siemens': [4.743574484899, 12.913499081063],
    'ceda_slope_real_micro_siemens': [3.2444288158784063, 1.821951365148186],
    'rmssd_percentile_0595': [34.1895276671302, 23.783359512525266],
    'sdnn_percentile_0595': [45.335573726241854, 24.38601160405501],
    'msa_probability': [48.1172590194961, 14.292898676874556],
    'hrv_percent_good': [0.2714810920080538, 0.2762414786979745],
    'hrv_rr_80th_percentile_mean': [828.8905850347666, 108.40428688789727],
    'hrv_rr_20th_percentile_mean': [734.5942838543058, 88.41269789220864],
    'hrv_rr_median': [780.925540250376, 94.86837708152842],
    'hrv_rr_mean': [785.7749142736874, 90.44585649648346],
    'hr_at_rest_mean': [82.86923994290905, 10.867752252500274],
    'skin_temperature_magnitude': [31.469650973107296, 1.7002512792231534],
    'skin_temperature_slope': [0.2655571148317653, 17.266512596820043],
    'rr_med': [856.8304, 160.1181],
    'rr_mean': [865.3986, 148.2940],
    'rr_80th': [924.0026, 170.5833],
    'sdnn': [96.8096, 69.3244],
    'sdnn0595': [64.8003, 55.5850],
    'rmssd': [104.6121, 105.0135],
    'rmssd0595': [65.3421, 74.7831],
    'pnn20': [0.5667, 0.2623],
    'pnn30': [0.4520, 0.2787],
    'pnn50': [0.3114, 0.2749],
    'pnn100': [0.1693, 0.2289],
    'ShEnRR': [3.0582, 0.6673],
    'ShEnRRD': [2.7086, 0.8037],
    'LF': [1551.8376, 2399.4228],
    'HF': [757.2271, 1873.9239],
    'LF_HF': [4.1265, 4.5066],
    'VLF': [1303.3848, 1906.1017],
    'coherence': [0.1808, 0.1305],
    'spectralEn': [2.5255, 0.3931],
    'percent_good': [0.4846, 0.3439],
    'sleep_stage_awake': [0.0424, 0.1916],
    'sleep_stage_light': [0.0434, 0.2021],
    'sleep_stage_deep': [0.1855, 0.3830],
    'sleep_stage_rem': [0.0575, 0.2301],
    'spo2': [95.2019, 2.4646],
    'spo2_is_valid': [0.5305, 0.4991],
    'spo2_confidence': [56.6391, 42.1064],
    'spo2_coverage': [50.1251, 19.0971],
}

FEATURES_TO_NORMALIZE = [
    'HR',
    'eda_level_real',
    'leads_contact_counts',
    'steps',
    'jerk_auto',
    'log_energy',
    'covariance',
    'log_energy_ratio',
    'zero_crossing_std',
    'zero_crossing_avg',
    'axis_mean',
    'altim_std',
    'kurtosis',
    'sleep_coefficient',
    'wrist_temperatures',
    'rr_med',
    'sdnn0595',
    'rmssd0595',
    'pnn20',
    'coherence',
    'ShEnRR',
    'LF',
    'HF',
    'LF_HF',
    'VLF',
    'spectralEn',
    'percent_good',
    'sleep_stage_awake',
    'sleep_stage_light',
    'sleep_stage_deep',
    'sleep_stage_rem',
    'spo2',
    'spo2_confidence',
    'spo2_coverage',
]


# Grouped channels for caption generation.
channel_groups_selected = {
    'Heart': {
        'members': [
            'HR',
            # 'hr_at_rest_mean',  # high corr
            # 'hrv_rr_80th_percentile_mean',  # high corr
            # 'hrv_rr_20th_percentile_mean',  # high corr
            'rr_med',
            'ShEnRR',
            # 'hrv_shannon_entropy_rrd',  # high corr
            # 'rmssd_percentile_0595',  # high corr
            'sdnn0595',
        ],
        'names': [
            'heart rate',
            # 'hr at rest mean',
            # 'hrv rr 80th percentile',
            # 'hrv rr 20th percentile',
            'hrv rr',
            'hrv shannon entropy rr',
            # 'hrv shannon entropy rrd',
            # 'rmssd percentile mean',
            'sdnn percentile',
        ],
    },
    'Activity': {
        'members': [
            'steps',
            'jerk_auto',
            'log_energy',
            # 'covariance',
            # 'log_energy_ratio',
            # 'zero_crossing_std',
            # 'zero_crossing_avg',
            # 'axis_mean',
            # 'altim_std',
            'kurtosis',
        ],
        'names': [
            'steps',
            'jerk',
            'log energy',
            # 'covariance',
            # 'log energy ratio',
            # 'zero crossing std',
            # 'zero crossing avg',
            # 'axis mean',
            # 'altim std',
            'kurtosis',
        ],
    },
    'Sleep': {
        'members': ['sleep_coefficient'],
        'names': ['sleep coefficient'],
    },
    'EDA': {
        'members': [
            'eda_level_real',
            # 'leads_contact_counts',
            # 'ceda_slope_real_micro_siemens',
            'wrist_temperatures',
            'wrist_temperatures',
        ],
        'names': [
            'eda level',
            # 'leads contact counts',
            # 'ceda slope real micro siemens',
            'skin temperature slope',
            'wrist temperatures',
        ],
    },
}

channel_groups_random = {
    'Heart': {
        'members': [
            # 'HR',
            'hr_at_rest_mean',  # high corr
            'rr_80th',  # high corr
            'rr_mean',  # high corr
            # 'hrv_rr_median',
            # 'hrv_shannon_entropy_rr',
            'ShEnRRD',  # high corr
            'rmssd0595',  # high corr
            # 'sdnn_percentile_0595',
        ],
        'names': [
            # 'heart rate',
            'hr at rest mean',
            'hrv rr 80th percentile',
            'hrv rr 20th percentile',
            # 'hrv rr',
            # 'hrv shannon entropy rr',
            'hrv shannon entropy rrd',
            'rmssd percentile mean',
            # 'sdnn percentile',
        ],
    },
    'Activity': {
        'members': [
            # 'steps',
            # 'jerk_auto',
            # 'log_energy',
            'covariance',
            'log_energy_ratio',
            'zero_crossing_std',
            'zero_crossing_avg',
            'axis_mean',
            'altim_std',
            # 'kurtosis',
        ],
        'names': [
            # 'steps',
            # 'jerk',
            # 'log energy',
            'covariance',
            'log energy ratio',
            'zero crossing std',
            'zero crossing avg',
            'axis mean',
            'altim std',
            # 'kurtosis',
        ],
    },
    'EDA': {
        'members': [
            # 'eda_level_real',
            'leads_contact_counts',
            'eda_level_real',
            # 'skin_temperature_slope',
            # 'wrist_temperatures',
        ],
        'names': [
            # 'eda level',
            'leads contact counts',
            'ceda slope real micro siemens',
            # 'skin temperature slope',
            # 'wrist temperatures',
        ],
    },
}

random_choose_nums = {
    'Heart': 2,
    'Activity': 1,
    'EDA': 1,
}

