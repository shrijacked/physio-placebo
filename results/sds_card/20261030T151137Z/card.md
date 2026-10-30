# SDS report card

Paradigm C is Qwen2.5-VL on WESAD and MAUS only. The grid is not a full 3 paradigms x 2 models x 3 datasets.

Primary numbers are the filed macro-F1 SDS. The secondary layer is post-registration (addendum 2026-10-30).

## Primary SDS

| Model | Dataset | View | Rung | M surrogate | Delta | Paired delta | Wilcoxon p | SDS |
|---|---|---|---|---:|---:|---|---:|---|
| qwen3_8b | wesad | A uniform_stride | gaussian | 0.466 | -0.039 | -0.040 [-0.100, 0.019] | 0.1909 | undefined (validity gate) |
| qwen3_8b | wesad | A uniform_stride | iaaft keeps level and spectrum (see ledger) | 0.385 | 0.042 | 0.041 [-0.013, 0.094] | 0.2734 | undefined (validity gate) |
| qwen3_8b | wesad | A uniform_stride | subject_perm_broken | 0.414 | 0.013 | 0.012 [-0.039, 0.062] | 0.8394 | undefined (validity gate) |
| qwen3_8b | wesad | A uniform_stride | flatline | 0.412 | 0.015 | 0.014 [-0.021, 0.051] | 0.8394 | undefined (validity gate) |
| qwen3_8b | wesad | A uniform_stride | deleted | 0.412 | 0.015 | 0.014 [-0.021, 0.051] | 0.8394 | undefined (validity gate) |
| qwen3_8b | wesad | A uniform_stride | subject_perm_preserved | 0.454 | -0.027 | -0.028 [-0.074, 0.014] | 0.1677 | descriptive, not a null |
| qwen3_8b | wesad | A bin_mean | gaussian | 0.482 | 0.113 | 0.116 [0.048, 0.180] | 0.0171 | 0.644 |
| qwen3_8b | wesad | A bin_mean | iaaft keeps level and spectrum (see ledger) | 0.534 | 0.060 | 0.064 [0.010, 0.121] | 0.0574 | 0.344 |
| qwen3_8b | wesad | A bin_mean | subject_perm_broken | 0.516 | 0.079 | 0.082 [0.026, 0.145] | 0.0803 | 0.450 |
| qwen3_8b | wesad | A bin_mean | flatline | 0.412 | 0.183 | 0.186 [0.141, 0.239] | 0.0002 | 1.041 |
| qwen3_8b | wesad | A bin_mean | deleted | 0.412 | 0.183 | 0.186 [0.141, 0.239] | 0.0002 | 1.041 |
| qwen3_8b | wesad | A bin_mean | subject_perm_preserved | 0.627 | -0.032 | -0.029 [-0.087, 0.037] | 0.3396 | descriptive, not a null |
| qwen3_8b | wesad | B none | gaussian | 0.292 | 0.292 | 0.288 [0.159, 0.413] | 0.0017 | 1.773 |
| qwen3_8b | wesad | B none | iaaft keeps level and spectrum (see ledger) | 0.704 | -0.120 | -0.124 [-0.218, -0.032] | 0.0342 | -0.732 |
| qwen3_8b | wesad | B none | subject_perm_broken | 0.454 | 0.130 | 0.126 [-0.005, 0.256] | 0.1677 | 0.787 |
| qwen3_8b | wesad | B none | flatline | 0.412 | 0.172 | 0.168 [0.048, 0.282] | 0.0574 | 1.044 |
| qwen3_8b | wesad | B none | deleted | 0.412 | 0.172 | 0.168 [0.048, 0.282] | 0.0574 | 1.044 |
| qwen3_8b | wesad | B none | subject_perm_preserved | 0.828 | -0.244 | -0.248 [-0.385, -0.118] | 0.0105 | descriptive, not a null |
| qwen3_8b | maus | A uniform_stride | gaussian | 0.400 | 0.067 | 0.064 [0.035, 0.097] | 0.0042 | 1.047 |
| qwen3_8b | maus | A uniform_stride | iaaft keeps level and spectrum (see ledger) | 0.476 | -0.009 | -0.012 [-0.053, 0.024] | 0.5461 | -0.137 |
| qwen3_8b | maus | A uniform_stride | subject_perm_broken | 0.478 | -0.011 | -0.015 [-0.055, 0.032] | 0.2024 | -0.177 |
| qwen3_8b | maus | A uniform_stride | flatline | 0.250 | 0.217 | 0.214 [0.185, 0.247] | 0.0000 | 3.372 |
| qwen3_8b | maus | A uniform_stride | deleted | 0.250 | 0.217 | 0.214 [0.185, 0.247] | 0.0000 | 3.372 |
| qwen3_8b | maus | A uniform_stride | subject_perm_preserved | 0.445 | 0.022 | 0.019 [-0.020, 0.061] | 0.5958 | descriptive, not a null |
| qwen3_8b | maus | A bin_mean | gaussian | 0.411 | 0.072 | 0.073 [0.039, 0.109] | 0.0124 | 0.898 |
| qwen3_8b | maus | A bin_mean | iaaft keeps level and spectrum (see ledger) | 0.488 | -0.005 | -0.004 [-0.049, 0.037] | 0.8405 | -0.065 |
| qwen3_8b | maus | A bin_mean | subject_perm_broken | 0.499 | -0.016 | -0.015 [-0.058, 0.024] | 0.3300 | -0.194 |
| qwen3_8b | maus | A bin_mean | flatline | 0.250 | 0.233 | 0.234 [0.206, 0.264] | 0.0001 | 2.894 |
| qwen3_8b | maus | A bin_mean | deleted | 0.250 | 0.233 | 0.234 [0.206, 0.264] | 0.0001 | 2.894 |
| qwen3_8b | maus | A bin_mean | subject_perm_preserved | 0.479 | 0.004 | 0.005 [-0.037, 0.050] | 0.7782 | descriptive, not a null |
| qwen3_8b | maus | B none | gaussian | 0.494 | -0.007 | -0.009 [-0.059, 0.039] | 0.2305 | -0.088 |
| qwen3_8b | maus | B none | iaaft keeps level and spectrum (see ledger) | 0.515 | -0.028 | -0.030 [-0.085, 0.026] | 0.2954 | -0.334 |
| qwen3_8b | maus | B none | subject_perm_broken | 0.495 | -0.008 | -0.009 [-0.062, 0.038] | 0.1769 | -0.091 |
| qwen3_8b | maus | B none | flatline | 0.250 | 0.237 | 0.235 [0.188, 0.279] | 0.0000 | 2.814 |
| qwen3_8b | maus | B none | deleted | 0.250 | 0.237 | 0.235 [0.188, 0.279] | 0.0000 | 2.814 |
| qwen3_8b | maus | B none | subject_perm_preserved | 0.499 | -0.012 | -0.013 [-0.075, 0.040] | 0.1536 | descriptive, not a null |
| qwen3_8b | cogwear | A uniform_stride | gaussian | 0.400 | 0.176 | 0.185 [0.049, 0.298] | 0.2188 | undefined (validity gate) |
| qwen3_8b | cogwear | A uniform_stride | iaaft keeps level and spectrum (see ledger) | 0.420 | 0.157 | 0.165 [0.052, 0.266] | 0.1562 | undefined (validity gate) |
| qwen3_8b | cogwear | A uniform_stride | subject_perm_broken | 0.644 | -0.067 | -0.059 [-0.215, 0.075] | 0.1953 | undefined (validity gate) |
| qwen3_8b | cogwear | A uniform_stride | flatline | 0.244 | 0.333 | 0.342 [0.215, 0.442] | 0.0078 | undefined (validity gate) |
| qwen3_8b | cogwear | A uniform_stride | deleted | 0.244 | 0.333 | 0.342 [0.215, 0.442] | 0.0078 | undefined (validity gate) |
| qwen3_8b | cogwear | A uniform_stride | subject_perm_preserved | 0.647 | -0.070 | -0.061 [-0.193, 0.041] | 0.3828 | descriptive, not a null |
| qwen3_8b | cogwear | A bin_mean | gaussian | 0.430 | -0.034 | -0.034 [-0.100, 0.013] | 0.6250 | undefined (validity gate) |
| qwen3_8b | cogwear | A bin_mean | iaaft keeps level and spectrum (see ledger) | 0.510 | -0.114 | -0.114 [-0.257, 0.017] | 0.3125 | undefined (validity gate) |
| qwen3_8b | cogwear | A bin_mean | subject_perm_broken | 0.521 | -0.125 | -0.125 [-0.273, 0.008] | 0.1484 | undefined (validity gate) |
| qwen3_8b | cogwear | A bin_mean | flatline | 0.404 | -0.008 | -0.008 [-0.025, 0.000] | 1.0000 | undefined (validity gate) |
| qwen3_8b | cogwear | A bin_mean | deleted | 0.244 | 0.152 | 0.152 [0.125, 0.185] | 0.0078 | undefined (validity gate) |
| qwen3_8b | cogwear | A bin_mean | subject_perm_preserved | 0.398 | -0.002 | -0.002 [-0.066, 0.059] | 0.4375 | descriptive, not a null |
| qwen3_8b | cogwear | B none | gaussian | 0.444 | 0.031 | 0.031 [-0.078, 0.100] | 0.2500 | undefined (validity gate) |
| qwen3_8b | cogwear | B none | iaaft keeps level and spectrum (see ledger) | 0.474 | 0.001 | 0.001 [-0.037, 0.037] | 0.7500 | undefined (validity gate) |
| qwen3_8b | cogwear | B none | subject_perm_broken | 0.507 | -0.032 | -0.032 [-0.173, 0.060] | 0.1094 | undefined (validity gate) |
| qwen3_8b | cogwear | B none | flatline | 0.244 | 0.231 | 0.231 [0.159, 0.256] | 0.0625 | undefined (validity gate) |
| qwen3_8b | cogwear | B none | deleted | 0.244 | 0.231 | 0.231 [0.159, 0.256] | 0.0625 | undefined (validity gate) |
| qwen3_8b | cogwear | B none | subject_perm_preserved | 0.456 | 0.019 | 0.019 [-0.095, 0.068] | 0.1484 | descriptive, not a null |
| llama31_8b | wesad | A uniform_stride | gaussian | 0.345 | -0.012 | -0.005 [-0.055, 0.052] | 0.5879 | undefined (validity gate) |
| llama31_8b | wesad | A uniform_stride | iaaft keeps level and spectrum (see ledger) | 0.347 | -0.014 | -0.007 [-0.044, 0.032] | 0.4548 | undefined (validity gate) |
| llama31_8b | wesad | A uniform_stride | subject_perm_broken | 0.391 | -0.059 | -0.052 [-0.083, -0.010] | 0.0171 | undefined (validity gate) |
| llama31_8b | wesad | A uniform_stride | flatline | 0.230 | 0.103 | 0.110 [0.076, 0.149] | 0.0002 | undefined (validity gate) |
| llama31_8b | wesad | A uniform_stride | deleted | 0.412 | -0.079 | -0.072 [-0.106, -0.033] | 0.0017 | undefined (validity gate) |
| llama31_8b | wesad | A uniform_stride | subject_perm_preserved | 0.335 | -0.002 | 0.005 [-0.044, 0.056] | 0.8394 | descriptive, not a null |
| llama31_8b | wesad | A bin_mean | gaussian | 0.230 | 0.000 | 0.000 [0.000, 0.000] |  | undefined (validity gate) |
| llama31_8b | wesad | A bin_mean | iaaft keeps level and spectrum (see ledger) | 0.230 | 0.000 | 0.000 [0.000, 0.000] |  | undefined (validity gate) |
| llama31_8b | wesad | A bin_mean | subject_perm_broken | 0.230 | 0.000 | 0.000 [0.000, 0.000] |  | undefined (validity gate) |
| llama31_8b | wesad | A bin_mean | flatline | 0.230 | 0.000 | 0.000 [0.000, 0.000] |  | undefined (validity gate) |
| llama31_8b | wesad | A bin_mean | deleted | 0.230 | 0.000 | 0.000 [0.000, 0.000] |  | undefined (validity gate) |
| llama31_8b | wesad | A bin_mean | subject_perm_preserved | 0.230 | 0.000 | 0.000 [0.000, 0.000] |  | descriptive, not a null |
| llama31_8b | wesad | B none | gaussian | 0.230 | 0.053 | 0.048 [0.014, 0.085] | 0.0156 | undefined (validity gate) |
| llama31_8b | wesad | B none | iaaft keeps level and spectrum (see ledger) | 0.230 | 0.053 | 0.048 [0.014, 0.085] | 0.0156 | undefined (validity gate) |
| llama31_8b | wesad | B none | subject_perm_broken | 0.230 | 0.053 | 0.048 [0.014, 0.085] | 0.0156 | undefined (validity gate) |
| llama31_8b | wesad | B none | flatline | 0.230 | 0.053 | 0.048 [0.014, 0.085] | 0.0156 | undefined (validity gate) |
| llama31_8b | wesad | B none | deleted | 0.412 | -0.129 | -0.134 [-0.168, -0.097] | 0.0007 | undefined (validity gate) |
| llama31_8b | wesad | B none | subject_perm_preserved | 0.230 | 0.053 | 0.048 [0.014, 0.085] | 0.0156 | descriptive, not a null |
| llama31_8b | maus | A uniform_stride | gaussian | 0.426 | -0.013 | -0.010 [-0.032, 0.012] | 0.4721 | undefined (validity gate) |
| llama31_8b | maus | A uniform_stride | iaaft keeps level and spectrum (see ledger) | 0.399 | 0.014 | 0.017 [0.005, 0.029] | 0.1369 | undefined (validity gate) |
| llama31_8b | maus | A uniform_stride | subject_perm_broken | 0.423 | -0.010 | -0.007 [-0.029, 0.016] | 0.5343 | undefined (validity gate) |
| llama31_8b | maus | A uniform_stride | flatline | 0.250 | 0.163 | 0.166 [0.155, 0.178] | 0.0001 | undefined (validity gate) |
| llama31_8b | maus | A uniform_stride | deleted | 0.250 | 0.163 | 0.166 [0.155, 0.178] | 0.0001 | undefined (validity gate) |
| llama31_8b | maus | A uniform_stride | subject_perm_preserved | 0.419 | -0.006 | -0.003 [-0.021, 0.016] | 0.2848 | descriptive, not a null |
| llama31_8b | maus | A bin_mean | gaussian | 0.400 | 0.000 | 0.000 [0.000, 0.000] |  | undefined (validity gate) |
| llama31_8b | maus | A bin_mean | iaaft keeps level and spectrum (see ledger) | 0.400 | 0.000 | 0.000 [0.000, 0.000] |  | undefined (validity gate) |
| llama31_8b | maus | A bin_mean | subject_perm_broken | 0.400 | 0.000 | 0.000 [0.000, 0.000] |  | undefined (validity gate) |
| llama31_8b | maus | A bin_mean | flatline | 0.400 | 0.000 | 0.000 [0.000, 0.000] |  | undefined (validity gate) |
| llama31_8b | maus | A bin_mean | deleted | 0.250 | 0.150 | 0.150 [0.150, 0.150] | 0.0000 | undefined (validity gate) |
| llama31_8b | maus | A bin_mean | subject_perm_preserved | 0.400 | 0.000 | 0.000 [0.000, 0.000] |  | descriptive, not a null |
| llama31_8b | maus | B none | gaussian | 0.499 | -0.099 | -0.099 [-0.131, -0.064] | 0.0017 | undefined (validity gate) |
| llama31_8b | maus | B none | iaaft keeps level and spectrum (see ledger) | 0.400 | 0.000 | 0.000 [0.000, 0.000] |  | undefined (validity gate) |
| llama31_8b | maus | B none | subject_perm_broken | 0.400 | 0.000 | 0.000 [0.000, 0.000] |  | undefined (validity gate) |
| llama31_8b | maus | B none | flatline | 0.400 | 0.000 | 0.000 [0.000, 0.000] |  | undefined (validity gate) |
| llama31_8b | maus | B none | deleted | 0.250 | 0.150 | 0.150 [0.150, 0.150] | 0.0000 | undefined (validity gate) |
| llama31_8b | maus | B none | subject_perm_preserved | 0.403 | -0.003 | -0.003 [-0.010, 0.000] | 0.3173 | descriptive, not a null |
| llama31_8b | cogwear | A uniform_stride | gaussian | 0.404 | 0.000 | 0.000 [0.000, 0.000] |  | undefined (validity gate) |
| llama31_8b | cogwear | A uniform_stride | iaaft keeps level and spectrum (see ledger) | 0.404 | 0.000 | 0.000 [0.000, 0.000] |  | undefined (validity gate) |
| llama31_8b | cogwear | A uniform_stride | subject_perm_broken | 0.404 | 0.000 | 0.000 [0.000, 0.000] |  | undefined (validity gate) |
| llama31_8b | cogwear | A uniform_stride | flatline | 0.404 | 0.000 | 0.000 [0.000, 0.000] |  | undefined (validity gate) |
| llama31_8b | cogwear | A uniform_stride | deleted | 0.244 | 0.160 | 0.160 [0.129, 0.194] | 0.0078 | undefined (validity gate) |
| llama31_8b | cogwear | A uniform_stride | subject_perm_preserved | 0.404 | 0.000 | 0.000 [0.000, 0.000] |  | descriptive, not a null |
| llama31_8b | cogwear | A bin_mean | gaussian | 0.266 | 0.174 | 0.174 [0.123, 0.230] | 0.0078 | undefined (validity gate) |
| llama31_8b | cogwear | A bin_mean | iaaft keeps level and spectrum (see ledger) | 0.444 | -0.004 | -0.004 [-0.175, 0.159] | 0.5781 | undefined (validity gate) |
| llama31_8b | cogwear | A bin_mean | subject_perm_broken | 0.400 | 0.040 | 0.040 [0.000, 0.101] | 0.5000 | undefined (validity gate) |
| llama31_8b | cogwear | A bin_mean | flatline | 0.404 | 0.036 | 0.036 [0.000, 0.099] | 1.0000 | undefined (validity gate) |
| llama31_8b | cogwear | A bin_mean | deleted | 0.244 | 0.196 | 0.196 [0.147, 0.249] | 0.0078 | undefined (validity gate) |
| llama31_8b | cogwear | A bin_mean | subject_perm_preserved | 0.410 | 0.030 | 0.030 [-0.077, 0.126] | 0.3750 | descriptive, not a null |
| llama31_8b | cogwear | B none | gaussian | 0.404 | 0.075 | 0.075 [-0.031, 0.229] | 1.0000 | undefined (validity gate) |
| llama31_8b | cogwear | B none | iaaft keeps level and spectrum (see ledger) | 0.496 | -0.017 | -0.017 [-0.139, 0.146] | 1.0000 | undefined (validity gate) |
| llama31_8b | cogwear | B none | subject_perm_broken | 0.400 | 0.079 | 0.079 [-0.027, 0.231] | 1.0000 | undefined (validity gate) |
| llama31_8b | cogwear | B none | flatline | 0.404 | 0.075 | 0.075 [-0.031, 0.229] | 1.0000 | undefined (validity gate) |
| llama31_8b | cogwear | B none | deleted | 0.244 | 0.235 | 0.235 [0.124, 0.412] | 0.0078 | undefined (validity gate) |
| llama31_8b | cogwear | B none | subject_perm_preserved | 0.440 | 0.039 | 0.039 [-0.111, 0.212] | 0.8750 | descriptive, not a null |
| qwen25_vl | wesad | C none | gaussian | 0.412 | -0.000 | -0.000 [-0.001, 0.000] | 1.0000 | undefined (validity gate) |
| qwen25_vl | wesad | C none | iaaft keeps level and spectrum (see ledger) | 0.301 | 0.111 | 0.111 [0.069, 0.148] | 0.0005 | undefined (validity gate) |
| qwen25_vl | wesad | C none | subject_perm_broken | 0.425 | -0.014 | -0.014 [-0.033, 0.002] | 0.4102 | undefined (validity gate) |
| qwen25_vl | wesad | C none | flatline | 0.412 | -0.000 | -0.000 [-0.001, 0.000] | 1.0000 | undefined (validity gate) |
| qwen25_vl | wesad | C none | deleted | 0.412 | -0.000 | -0.000 [-0.001, 0.000] | 1.0000 | undefined (validity gate) |
| qwen25_vl | wesad | C none | subject_perm_preserved | 0.652 | -0.240 | -0.240 [-0.275, -0.207] | 0.0002 | descriptive, not a null |
| qwen25_vl | maus | C none | gaussian | 0.250 | 0.150 | 0.150 [0.149, 0.150] | 0.0000 | undefined (validity gate) |
| qwen25_vl | maus | C none | iaaft keeps level and spectrum (see ledger) | 0.400 | -0.000 | -0.000 [-0.001, 0.000] | 0.3173 | undefined (validity gate) |
| qwen25_vl | maus | C none | subject_perm_broken | 0.467 | -0.068 | -0.068 [-0.097, -0.039] | 0.0017 | undefined (validity gate) |
| qwen25_vl | maus | C none | flatline | 0.250 | 0.150 | 0.150 [0.149, 0.150] | 0.0000 | undefined (validity gate) |
| qwen25_vl | maus | C none | deleted | 0.250 | 0.150 | 0.150 [0.149, 0.150] | 0.0000 | undefined (validity gate) |
| qwen25_vl | maus | C none | subject_perm_preserved | 0.400 | -0.000 | -0.000 [-0.001, 0.000] | 0.3173 | descriptive, not a null |

## Classical delta

| Model | Dataset | View | LLM | Logreg | Delta |
|---|---|---|---:|---:|---:|
| qwen3_8b | wesad | A uniform_stride | 0.427 | 0.918 | -0.491 |
| qwen3_8b | wesad | A bin_mean | 0.595 | 0.918 | -0.324 |
| qwen3_8b | wesad | B none | 0.584 | 0.918 | -0.334 |
| qwen3_8b | maus | A uniform_stride | 0.467 | 0.444 | 0.023 |
| qwen3_8b | maus | A bin_mean | 0.483 | 0.444 | 0.039 |
| qwen3_8b | maus | B none | 0.487 | 0.444 | 0.043 |
| qwen3_8b | cogwear | A uniform_stride | 0.576 | 0.569 | 0.007 |
| qwen3_8b | cogwear | A bin_mean | 0.396 | 0.569 | -0.173 |
| qwen3_8b | cogwear | B none | 0.475 | 0.569 | -0.094 |
| llama31_8b | wesad | A uniform_stride | 0.333 | 0.918 | -0.585 |
| llama31_8b | wesad | A bin_mean | 0.230 | 0.918 | -0.688 |
| llama31_8b | wesad | B none | 0.283 | 0.918 | -0.635 |
| llama31_8b | maus | A uniform_stride | 0.413 | 0.444 | -0.031 |
| llama31_8b | maus | A bin_mean | 0.400 | 0.444 | -0.044 |
| llama31_8b | maus | B none | 0.400 | 0.444 | -0.044 |
| llama31_8b | cogwear | A uniform_stride | 0.404 | 0.569 | -0.165 |
| llama31_8b | cogwear | A bin_mean | 0.440 | 0.569 | -0.129 |
| llama31_8b | cogwear | B none | 0.479 | 0.569 | -0.090 |
| qwen25_vl | wesad | C none | 0.412 | 0.918 | -0.506 |
| qwen25_vl | maus | C none | 0.400 | 0.444 | -0.044 |

## Verbalizer panel

Qwen MAUS variant (i) is 0.485 against a matched floor of 0.499 (p95 0.519). P2 holds on MAUS only under the filed classical floor.

| Model | Dataset | As written | Words removed | Words flipped |
|---|---|---:|---:|---:|
| llama31_8b | wesad | 0.278 | 0.230 | 0.230 |
| llama31_8b | maus | 0.400 | 0.400 | 0.503 |
| llama31_8b | cogwear | 0.479 | 0.404 | 0.404 |
| qwen3_8b | wesad | 0.581 | 0.554 | 0.313 |
| qwen3_8b | maus | 0.485 | 0.399 | 0.507 |
| qwen3_8b | cogwear | 0.475 | 0.542 | 0.597 |

Adjective vote (post-registration, no language model).

| Model | Dataset | Vote macro-F1 | Vote BA | Vote AUROC | Agreement | Spearman |
|---|---|---:|---:|---:|---:|---:|
| qwen3_8b | wesad | 0.736 | 0.778 | 0.857 | 0.693 | 0.842 |
| qwen3_8b | maus | 0.448 | 0.461 | 0.482 | 0.835 | 0.884 |
| qwen3_8b | cogwear | 0.421 | 0.437 | 0.388 | 0.811 | 0.941 |
| llama31_8b | wesad | 0.736 | 0.778 | 0.857 | 0.482 | 0.710 |
| llama31_8b | maus | 0.448 | 0.461 | 0.482 | 0.485 | 0.469 |
| llama31_8b | cogwear | 0.421 | 0.437 | 0.388 | 0.544 | 0.931 |

## Secondary layer

post-registration (addendum 2026-10-30).

| Model | Dataset | View | Rung | BA | AUROC | Floor | SDS_BA | SDS_AUC | Holm |
|---|---|---|---|---:|---:|---:|---|---|---:|
| qwen3_8b | wesad | A uniform_stride | gaussian | 0.497 | 0.487 | 0.427 | undefined (validity gate) | undefined (validity gate) | 1.0000 |
| qwen3_8b | wesad | A uniform_stride | iaaft | 0.497 | 0.487 | 0.427 | undefined (validity gate) | undefined (validity gate) | 1.0000 |
| qwen3_8b | wesad | A uniform_stride | subject_perm_broken | 0.497 | 0.487 | 0.427 | undefined (validity gate) | undefined (validity gate) | 1.0000 |
| qwen3_8b | wesad | A uniform_stride | flatline | 0.497 | 0.487 | 0.427 | undefined (validity gate) | undefined (no item-level variation) | 1.0000 |
| qwen3_8b | wesad | A uniform_stride | deleted | 0.497 | 0.487 | 0.427 | undefined (validity gate) | undefined (no item-level variation) | 1.0000 |
| qwen3_8b | wesad | A uniform_stride | subject_perm_preserved | 0.497 | 0.487 | 0.427 | undefined (validity gate) | undefined (validity gate) |  |
| qwen3_8b | wesad | A bin_mean | gaussian | 0.599 | 0.641 | 0.500 | 1.182 | 1.377 | 0.8374 |
| qwen3_8b | wesad | A bin_mean | iaaft | 0.599 | 0.641 | 0.500 | 0.658 | 0.645 | 1.0000 |
| qwen3_8b | wesad | A bin_mean | subject_perm_broken | 0.599 | 0.641 | 0.500 | 0.841 | 0.887 | 1.0000 |
| qwen3_8b | wesad | A bin_mean | flatline | 0.599 | 0.641 | 0.500 | 1.000 | undefined (no item-level variation) | 0.0173 |
| qwen3_8b | wesad | A bin_mean | deleted | 0.599 | 0.641 | 0.500 | 1.000 | undefined (no item-level variation) | 0.0173 |
| qwen3_8b | wesad | A bin_mean | subject_perm_preserved | 0.599 | 0.641 | 0.500 | -0.291 | -0.276 |  |
| qwen3_8b | wesad | B none | gaussian | 0.698 | 0.937 | 0.415 | 1.053 | 1.041 | 0.1127 |
| qwen3_8b | wesad | B none | iaaft | 0.698 | 0.937 | 0.415 | -0.320 | 0.149 | 1.0000 |
| qwen3_8b | wesad | B none | subject_perm_broken | 0.698 | 0.937 | 0.415 | 0.911 | 0.990 | 1.0000 |
| qwen3_8b | wesad | B none | flatline | 0.698 | 0.937 | 0.415 | 1.000 | undefined (no item-level variation) | 1.0000 |
| qwen3_8b | wesad | B none | deleted | 0.698 | 0.937 | 0.415 | 1.000 | undefined (no item-level variation) | 1.0000 |
| qwen3_8b | wesad | B none | subject_perm_preserved | 0.698 | 0.937 | 0.415 | -0.617 | 0.038 |  |
| qwen3_8b | maus | A uniform_stride | gaussian | 0.487 | 0.490 | 0.476 | undefined (validity gate) | undefined (validity gate) | 0.2617 |
| qwen3_8b | maus | A uniform_stride | iaaft | 0.487 | 0.490 | 0.476 | undefined (validity gate) | undefined (validity gate) | 1.0000 |
| qwen3_8b | maus | A uniform_stride | subject_perm_broken | 0.487 | 0.490 | 0.476 | undefined (validity gate) | undefined (validity gate) | 1.0000 |
| qwen3_8b | maus | A uniform_stride | flatline | 0.487 | 0.490 | 0.476 | undefined (validity gate) | undefined (validity gate) | 0.0002 |
| qwen3_8b | maus | A uniform_stride | deleted | 0.487 | 0.490 | 0.476 | undefined (validity gate) | undefined (no item-level variation) | 0.0002 |
| qwen3_8b | maus | A uniform_stride | subject_perm_preserved | 0.487 | 0.490 | 0.476 | undefined (validity gate) | undefined (validity gate) |  |
| qwen3_8b | maus | A bin_mean | gaussian | 0.500 | 0.497 | 0.484 | undefined (validity gate) | undefined (validity gate) | 0.6678 |
| qwen3_8b | maus | A bin_mean | iaaft | 0.500 | 0.497 | 0.484 | undefined (validity gate) | undefined (validity gate) | 1.0000 |
| qwen3_8b | maus | A bin_mean | subject_perm_broken | 0.500 | 0.497 | 0.484 | undefined (validity gate) | undefined (validity gate) | 1.0000 |
| qwen3_8b | maus | A bin_mean | flatline | 0.500 | 0.497 | 0.484 | undefined (validity gate) | undefined (validity gate) | 0.0065 |
| qwen3_8b | maus | A bin_mean | deleted | 0.500 | 0.497 | 0.484 | undefined (validity gate) | undefined (no item-level variation) | 0.0065 |
| qwen3_8b | maus | A bin_mean | subject_perm_preserved | 0.500 | 0.497 | 0.484 | undefined (validity gate) | undefined (validity gate) |  |
| qwen3_8b | maus | B none | gaussian | 0.486 | 0.497 | 0.499 | undefined (validity gate) | undefined (validity gate) | 1.0000 |
| qwen3_8b | maus | B none | iaaft | 0.486 | 0.497 | 0.499 | undefined (validity gate) | undefined (validity gate) | 1.0000 |
| qwen3_8b | maus | B none | subject_perm_broken | 0.486 | 0.497 | 0.499 | undefined (validity gate) | undefined (validity gate) | 1.0000 |
| qwen3_8b | maus | B none | flatline | 0.486 | 0.497 | 0.499 | undefined (validity gate) | undefined (no item-level variation) | 0.0003 |
| qwen3_8b | maus | B none | deleted | 0.486 | 0.497 | 0.499 | undefined (validity gate) | undefined (no item-level variation) | 0.0003 |
| qwen3_8b | maus | B none | subject_perm_preserved | 0.486 | 0.497 | 0.499 | undefined (validity gate) | undefined (validity gate) |  |
| qwen3_8b | cogwear | A uniform_stride | gaussian | 0.584 | 0.567 | 0.491 | undefined (validity gate) | undefined (validity gate) | 1.0000 |
| qwen3_8b | cogwear | A uniform_stride | iaaft | 0.584 | 0.567 | 0.491 | undefined (validity gate) | undefined (validity gate) | 1.0000 |
| qwen3_8b | cogwear | A uniform_stride | subject_perm_broken | 0.584 | 0.567 | 0.491 | undefined (validity gate) | undefined (validity gate) | 1.0000 |
| qwen3_8b | cogwear | A uniform_stride | flatline | 0.584 | 0.567 | 0.491 | undefined (validity gate) | undefined (no item-level variation) | 0.4766 |
| qwen3_8b | cogwear | A uniform_stride | deleted | 0.584 | 0.567 | 0.491 | undefined (validity gate) | undefined (no item-level variation) | 0.4766 |
| qwen3_8b | cogwear | A uniform_stride | subject_perm_preserved | 0.584 | 0.567 | 0.491 | undefined (validity gate) | undefined (validity gate) |  |
| qwen3_8b | cogwear | A bin_mean | gaussian | 0.484 | 0.574 | 0.416 | undefined (validity gate) | undefined (validity gate) | 1.0000 |
| qwen3_8b | cogwear | A bin_mean | iaaft | 0.484 | 0.574 | 0.416 | undefined (validity gate) | undefined (validity gate) | 1.0000 |
| qwen3_8b | cogwear | A bin_mean | subject_perm_broken | 0.484 | 0.574 | 0.416 | undefined (validity gate) | undefined (validity gate) | 1.0000 |
| qwen3_8b | cogwear | A bin_mean | flatline | 0.484 | 0.574 | 0.416 | undefined (validity gate) | undefined (no item-level variation) | 1.0000 |
| qwen3_8b | cogwear | A bin_mean | deleted | 0.484 | 0.574 | 0.416 | undefined (validity gate) | undefined (no item-level variation) | 0.4766 |
| qwen3_8b | cogwear | A bin_mean | subject_perm_preserved | 0.484 | 0.574 | 0.416 | undefined (validity gate) | undefined (validity gate) |  |
| qwen3_8b | cogwear | B none | gaussian | 0.475 | 0.417 | 0.466 | undefined (validity gate) | undefined (validity gate) | 1.0000 |
| qwen3_8b | cogwear | B none | iaaft | 0.475 | 0.417 | 0.466 | undefined (validity gate) | undefined (validity gate) | 1.0000 |
| qwen3_8b | cogwear | B none | subject_perm_broken | 0.475 | 0.417 | 0.466 | undefined (validity gate) | undefined (validity gate) | 1.0000 |
| qwen3_8b | cogwear | B none | flatline | 0.475 | 0.417 | 0.466 | undefined (validity gate) | undefined (no item-level variation) | 1.0000 |
| qwen3_8b | cogwear | B none | deleted | 0.475 | 0.417 | 0.466 | undefined (validity gate) | undefined (no item-level variation) | 1.0000 |
| qwen3_8b | cogwear | B none | subject_perm_preserved | 0.475 | 0.417 | 0.466 | undefined (validity gate) | undefined (validity gate) |  |
| llama31_8b | wesad | A uniform_stride | gaussian | 0.462 | 0.416 | 0.371 | undefined (validity gate) | undefined (validity gate) | 1.0000 |
| llama31_8b | wesad | A uniform_stride | iaaft | 0.462 | 0.416 | 0.371 | undefined (validity gate) | undefined (validity gate) | 1.0000 |
| llama31_8b | wesad | A uniform_stride | subject_perm_broken | 0.462 | 0.416 | 0.371 | undefined (validity gate) | undefined (validity gate) | 0.8374 |
| llama31_8b | wesad | A uniform_stride | flatline | 0.462 | 0.416 | 0.371 | undefined (validity gate) | undefined (no item-level variation) | 0.0173 |
| llama31_8b | wesad | A uniform_stride | deleted | 0.462 | 0.416 | 0.371 | undefined (validity gate) | undefined (no item-level variation) | 0.1127 |
| llama31_8b | wesad | A uniform_stride | subject_perm_preserved | 0.462 | 0.416 | 0.371 | undefined (validity gate) | undefined (validity gate) |  |
| llama31_8b | wesad | A bin_mean | gaussian | 0.500 | 0.588 | 0.230 | undefined (validity gate) | 1.214 |  |
| llama31_8b | wesad | A bin_mean | iaaft | 0.500 | 0.588 | 0.230 | undefined (validity gate) | 0.984 |  |
| llama31_8b | wesad | A bin_mean | subject_perm_broken | 0.500 | 0.588 | 0.230 | undefined (validity gate) | 0.959 |  |
| llama31_8b | wesad | A bin_mean | flatline | 0.500 | 0.588 | 0.230 | undefined (validity gate) | undefined (no item-level variation) |  |
| llama31_8b | wesad | A bin_mean | deleted | 0.500 | 0.588 | 0.230 | undefined (validity gate) | undefined (no item-level variation) |  |
| llama31_8b | wesad | A bin_mean | subject_perm_preserved | 0.500 | 0.588 | 0.230 | undefined (validity gate) | -0.034 |  |
| llama31_8b | wesad | B none | gaussian | 0.522 | 0.819 | 0.258 | 1.000 | 1.016 | 0.8281 |
| llama31_8b | wesad | B none | iaaft | 0.522 | 0.819 | 0.258 | 1.000 | 0.033 | 0.8281 |
| llama31_8b | wesad | B none | subject_perm_broken | 0.522 | 0.819 | 0.258 | 1.000 | 0.928 | 0.8281 |
| llama31_8b | wesad | B none | flatline | 0.522 | 0.819 | 0.258 | 1.000 | undefined (no item-level variation) | 0.8281 |
| llama31_8b | wesad | B none | deleted | 0.522 | 0.819 | 0.258 | 1.000 | undefined (no item-level variation) | 0.0491 |
| llama31_8b | wesad | B none | subject_perm_preserved | 0.522 | 0.819 | 0.258 | 1.000 | -0.140 |  |
| llama31_8b | maus | A uniform_stride | gaussian | 0.503 | 0.499 | 0.411 | undefined (validity gate) | undefined (validity gate) | 1.0000 |
| llama31_8b | maus | A uniform_stride | iaaft | 0.503 | 0.499 | 0.411 | undefined (validity gate) | undefined (validity gate) | 1.0000 |
| llama31_8b | maus | A uniform_stride | subject_perm_broken | 0.503 | 0.499 | 0.411 | undefined (validity gate) | undefined (validity gate) | 1.0000 |
| llama31_8b | maus | A uniform_stride | flatline | 0.503 | 0.499 | 0.411 | undefined (validity gate) | undefined (no item-level variation) | 0.0048 |
| llama31_8b | maus | A uniform_stride | deleted | 0.503 | 0.499 | 0.411 | undefined (validity gate) | undefined (no item-level variation) | 0.0048 |
| llama31_8b | maus | A uniform_stride | subject_perm_preserved | 0.503 | 0.499 | 0.411 | undefined (validity gate) | undefined (validity gate) |  |
| llama31_8b | maus | A bin_mean | gaussian | 0.500 | 0.481 | 0.400 | undefined (validity gate) | undefined (validity gate) |  |
| llama31_8b | maus | A bin_mean | iaaft | 0.500 | 0.481 | 0.400 | undefined (validity gate) | undefined (validity gate) |  |
| llama31_8b | maus | A bin_mean | subject_perm_broken | 0.500 | 0.481 | 0.400 | undefined (validity gate) | undefined (validity gate) |  |
| llama31_8b | maus | A bin_mean | flatline | 0.500 | 0.481 | 0.400 | undefined (validity gate) | undefined (no item-level variation) |  |
| llama31_8b | maus | A bin_mean | deleted | 0.500 | 0.481 | 0.400 | undefined (validity gate) | undefined (no item-level variation) | 0.0006 |
| llama31_8b | maus | A bin_mean | subject_perm_preserved | 0.500 | 0.481 | 0.400 | undefined (validity gate) | undefined (validity gate) |  |
| llama31_8b | maus | B none | gaussian | 0.500 | 0.509 | 0.400 | undefined (validity gate) | undefined (validity gate) | 0.1127 |
| llama31_8b | maus | B none | iaaft | 0.500 | 0.509 | 0.400 | undefined (validity gate) | undefined (validity gate) |  |
| llama31_8b | maus | B none | subject_perm_broken | 0.500 | 0.509 | 0.400 | undefined (validity gate) | undefined (validity gate) |  |
| llama31_8b | maus | B none | flatline | 0.500 | 0.509 | 0.400 | undefined (validity gate) | undefined (no item-level variation) |  |
| llama31_8b | maus | B none | deleted | 0.500 | 0.509 | 0.400 | undefined (validity gate) | undefined (no item-level variation) | 0.0006 |
| llama31_8b | maus | B none | subject_perm_preserved | 0.500 | 0.509 | 0.400 | undefined (validity gate) | undefined (validity gate) |  |
| llama31_8b | cogwear | A uniform_stride | gaussian | 0.500 | 0.540 | 0.404 | undefined (validity gate) | undefined (validity gate) |  |
| llama31_8b | cogwear | A uniform_stride | iaaft | 0.500 | 0.540 | 0.404 | undefined (validity gate) | undefined (validity gate) |  |
| llama31_8b | cogwear | A uniform_stride | subject_perm_broken | 0.500 | 0.540 | 0.404 | undefined (validity gate) | undefined (validity gate) |  |
| llama31_8b | cogwear | A uniform_stride | flatline | 0.500 | 0.540 | 0.404 | undefined (validity gate) | undefined (no item-level variation) |  |
| llama31_8b | cogwear | A uniform_stride | deleted | 0.500 | 0.540 | 0.404 | undefined (validity gate) | undefined (no item-level variation) | 0.4766 |
| llama31_8b | cogwear | A uniform_stride | subject_perm_preserved | 0.500 | 0.540 | 0.404 | undefined (validity gate) | undefined (validity gate) |  |
| llama31_8b | cogwear | A bin_mean | gaussian | 0.517 | 0.488 | 0.415 | undefined (validity gate) | undefined (validity gate) | 0.4766 |
| llama31_8b | cogwear | A bin_mean | iaaft | 0.517 | 0.488 | 0.415 | undefined (validity gate) | undefined (validity gate) | 1.0000 |
| llama31_8b | cogwear | A bin_mean | subject_perm_broken | 0.517 | 0.488 | 0.415 | undefined (validity gate) | undefined (validity gate) | 1.0000 |
| llama31_8b | cogwear | A bin_mean | flatline | 0.517 | 0.488 | 0.415 | undefined (validity gate) | undefined (no item-level variation) | 1.0000 |
| llama31_8b | cogwear | A bin_mean | deleted | 0.517 | 0.488 | 0.415 | undefined (validity gate) | undefined (no item-level variation) | 0.4766 |
| llama31_8b | cogwear | A bin_mean | subject_perm_preserved | 0.517 | 0.488 | 0.415 | undefined (validity gate) | undefined (validity gate) |  |
| llama31_8b | cogwear | B none | gaussian | 0.519 | 0.473 | 0.443 | undefined (validity gate) | undefined (validity gate) | 1.0000 |
| llama31_8b | cogwear | B none | iaaft | 0.519 | 0.473 | 0.443 | undefined (validity gate) | undefined (validity gate) | 1.0000 |
| llama31_8b | cogwear | B none | subject_perm_broken | 0.519 | 0.473 | 0.443 | undefined (validity gate) | undefined (validity gate) | 1.0000 |
| llama31_8b | cogwear | B none | flatline | 0.519 | 0.473 | 0.443 | undefined (validity gate) | undefined (no item-level variation) | 1.0000 |
| llama31_8b | cogwear | B none | deleted | 0.519 | 0.473 | 0.443 | undefined (validity gate) | undefined (no item-level variation) | 0.4766 |
| llama31_8b | cogwear | B none | subject_perm_preserved | 0.519 | 0.473 | 0.443 | undefined (validity gate) | undefined (validity gate) |  |
| qwen25_vl | wesad | C none | gaussian | 0.499 | 0.746 | 0.413 | undefined (validity gate) | undefined (no item-level variation) | 1.0000 |
| qwen25_vl | wesad | C none | iaaft | 0.499 | 0.746 | 0.413 | undefined (validity gate) | undefined (no item-level variation) | 0.0332 |
| qwen25_vl | wesad | C none | subject_perm_broken | 0.499 | 0.746 | 0.413 | undefined (validity gate) | undefined (no item-level variation) | 1.0000 |
| qwen25_vl | wesad | C none | flatline | 0.499 | 0.746 | 0.413 | undefined (validity gate) | undefined (no item-level variation) | 1.0000 |
| qwen25_vl | wesad | C none | deleted | 0.499 | 0.746 | 0.413 | undefined (validity gate) | undefined (no item-level variation) | 1.0000 |
| qwen25_vl | wesad | C none | subject_perm_preserved | 0.499 | 0.746 | 0.413 | undefined (validity gate) | -0.192 |  |
| qwen25_vl | maus | C none | gaussian | 0.499 | 0.497 | 0.401 | undefined (validity gate) | undefined (no item-level variation) | 0.0009 |
| qwen25_vl | maus | C none | iaaft | 0.499 | 0.497 | 0.401 | undefined (validity gate) | undefined (no item-level variation) | 1.0000 |
| qwen25_vl | maus | C none | subject_perm_broken | 0.499 | 0.497 | 0.401 | undefined (validity gate) | undefined (no item-level variation) | 0.1127 |
| qwen25_vl | maus | C none | flatline | 0.499 | 0.497 | 0.401 | undefined (validity gate) | undefined (no item-level variation) | 0.0009 |
| qwen25_vl | maus | C none | deleted | 0.499 | 0.497 | 0.401 | undefined (validity gate) | undefined (no item-level variation) | 0.0009 |
| qwen25_vl | maus | C none | subject_perm_preserved | 0.499 | 0.497 | 0.401 | undefined (validity gate) | undefined (no item-level variation) |  |

## Input-matched ceilings

post-registration (addendum 2026-10-30). Surrogate windows are not re-fit.

| Dataset | Ceiling | Scheme | Eval macro-F1 | Eval AUROC | All macro-F1 | All AUROC |
|---|---|---|---:|---:|---:|---:|
| wesad | hrv |  | 0.786 | 0.876 | 0.783 | 0.866 |
| wesad | prompt_stats | bin_mean | 0.805 | 0.917 | 0.786 | 0.907 |
| wesad | prompt_stats | uniform_stride | 0.688 | 0.730 | 0.693 | 0.731 |
| maus | hrv |  | 0.448 | 0.560 | 0.444 | 0.566 |
| maus | prompt_stats | bin_mean | 0.400 | 0.488 | 0.400 | 0.509 |
| maus | prompt_stats | uniform_stride | 0.400 | 0.480 | 0.400 | 0.500 |
| cogwear | hrv |  | 0.725 | 0.674 | 0.686 | 0.657 |
| cogwear | prompt_stats | bin_mean | 0.404 | 0.548 | 0.398 | 0.543 |
| cogwear | prompt_stats | uniform_stride | 0.421 | 0.508 | 0.411 | 0.528 |

## Determinism

| Model | Dataset | View | Locked | Re-score | Abs delta | Flag |
|---|---|---|---:|---:|---:|---|
| qwen3_8b | wesad | A uniform_stride | 0.427 | 0.426 | 0.001 | ok |
| qwen3_8b | wesad | A bin_mean | 0.595 | 0.598 | 0.003 | ok |
| qwen3_8b | wesad | B none | 0.584 | 0.581 | 0.003 | ok |
| qwen3_8b | maus | A uniform_stride | 0.467 | 0.464 | 0.003 | ok |
| qwen3_8b | maus | A bin_mean | 0.483 | 0.484 | 0.001 | ok |
| qwen3_8b | maus | B none | 0.487 | 0.485 | 0.001 | ok |
| qwen3_8b | cogwear | A uniform_stride | 0.576 | 0.585 | 0.009 | ok |
| qwen3_8b | cogwear | A bin_mean | 0.396 | 0.396 | 0.000 | ok |
| qwen3_8b | cogwear | B none | 0.475 | 0.475 | 0.000 | ok |
| llama31_8b | wesad | A uniform_stride | 0.333 | 0.340 | 0.007 | ok |
| llama31_8b | wesad | A bin_mean | 0.230 | 0.230 | 0.000 | ok |
| llama31_8b | wesad | B none | 0.283 | 0.278 | 0.005 | ok |
| llama31_8b | maus | A uniform_stride | 0.413 | 0.416 | 0.003 | ok |
| llama31_8b | maus | A bin_mean | 0.400 | 0.400 | 0.000 | ok |
| llama31_8b | maus | B none | 0.400 | 0.400 | 0.000 | ok |
| llama31_8b | cogwear | A uniform_stride | 0.404 | 0.404 | 0.000 | ok |
| llama31_8b | cogwear | A bin_mean | 0.440 | 0.440 | 0.000 | ok |
| llama31_8b | cogwear | B none | 0.479 | 0.479 | 0.000 | ok |
| qwen25_vl | wesad | C none | 0.412 | 0.412 | 0.000 | ok |
| qwen25_vl | maus | C none | 0.400 | 0.400 | 0.000 | ok |
