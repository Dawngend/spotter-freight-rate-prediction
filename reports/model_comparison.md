# Model Comparison

## Evaluation Design

Fit January through August and tune on September only. Select the lowest September rate MAE, with RMSE as tie-break. Freeze both model choices in outputs/model_choices.json before evaluating October. The October result refits through September. The two-month result keeps the January through August fit and evaluates September plus October together. It includes the tuning month and is not an independent holdout. October is evaluated after choices are frozen and never used to revise them. Stage 1 already described October distributions, so untouched here refers to model selection and prediction errors.

All rows are retained. RPM training targets use posted_rate / distance_used and predictions multiply by distance_used. The other target is log(posted_rate), inverted with exp without a smearing correction. Metrics are in rate units except percentage errors and R2. No target outlier removal or learned prediction clipping is used.

## Features and Search

Coordinates, air miles, the supplied distance, log distance, the road-to-air ratio (capped at 10), a flag for ratios above three, equipment, weight and calendar variables form the core inputs. The supplied distance is used as is in the features and as the RPM denominator (see Distance Policy). Negative weights become missing, with separate negative and missing flags. Weight medians are fitted per equipment with a global fallback. Market-only models add market_index and its missing flag. All medians, scaling and equipment encoders are fitted inside each fitting window. City strings are used for lookup and lane diagnostics only; load_id is used only to align outputs.

Ridge alpha candidates and histogram leaf-count candidates appear in the parameter part of each candidate name. Histogram models use squared or absolute loss, 200 iterations, learning rate 0.08, minimum leaf size 30 and L2 regularization 1. Early stopping is disabled so no random internal holdout influences the comparison. The fixed random seed is 42. Baselines use global or equipment median RPM times distance. Both baselines are signal-free by definition.

Quote-only and both-signal ablations are deliberately ineligible following conclusion (b) in quote_signal_regimes.md. The eligible ablations are no signals and market_index only. The same target and hyperparameter search is used for each eligible signal setting. A label-free regime detector (src/regime_detect.py, reports/quote_regime_detection.md) later read all ten development months correctly using only columns that also exist in validation, and found both validation months unresolved, like August, so quote_signal is noise in November and December. No regime-aware prediction file is produced and quote_signal stays excluded.

## September Selection

| index | candidate | rows | MAE | RMSE | MAPE_pct | MdAPE_pct | R2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | median / rpm / none / 0 | 4670 | 260.8321 | 673.3033 | 11.4477 | 8.3686 | 0.8046 |
| 1 | equipment_median / rpm / none / 0 | 4670 | 227.3228 | 657.0658 | 10.0364 | 6.9387 | 0.8139 |
| 2 | ridge / rpm / none / 1 | 4670 | 133.8957 | 617.1020 | 5.6193 | 3.2407 | 0.8359 |
| 3 | ridge / rpm / none / 100 | 4670 | 168.3405 | 623.0467 | 7.0271 | 4.8387 | 0.8327 |
| 4 | hist_squared / rpm / none / 15 | 4670 | 119.7881 | 618.3648 | 4.9367 | 2.4302 | 0.8352 |
| 5 | hist_squared / rpm / none / 31 | 4670 | 126.7171 | 617.3466 | 5.3121 | 2.6875 | 0.8358 |
| 6 | hist_absolute / rpm / none / 15 | 4670 | 102.9204 | 616.4266 | 4.1847 | 1.7441 | 0.8363 |
| 7 | hist_absolute / rpm / none / 31 | 4670 | 102.7044 | 616.1333 | 4.1753 | 1.7657 | 0.8364 |
| 8 | ridge / log_rate / none / 1 | 4670 | 113.7387 | 614.0001 | 4.7689 | 2.3437 | 0.8375 |
| 9 | ridge / log_rate / none / 100 | 4670 | 140.9492 | 617.8117 | 5.9256 | 3.6090 | 0.8355 |
| 10 | hist_squared / log_rate / none / 15 | 4670 | 113.2963 | 617.7599 | 4.6773 | 2.1859 | 0.8355 |
| 11 | hist_squared / log_rate / none / 31 | 4670 | 119.0678 | 618.1171 | 4.9565 | 2.4622 | 0.8354 |
| 12 | hist_absolute / log_rate / none / 15 | 4670 | 107.2523 | 617.2742 | 4.4460 | 1.9718 | 0.8358 |
| 13 | hist_absolute / log_rate / none / 31 | 4670 | 105.5688 | 615.6220 | 4.3537 | 1.9012 | 0.8367 |
| 14 | ridge / rpm / market / 1 | 4670 | 105.6186 | 617.0033 | 4.3590 | 1.9708 | 0.8359 |
| 15 | ridge / rpm / market / 100 | 4670 | 104.4431 | 616.4238 | 4.3306 | 1.9338 | 0.8363 |
| 16 | hist_squared / rpm / market / 15 | 4670 | 122.6177 | 620.6835 | 5.1015 | 2.6571 | 0.8340 |
| 17 | hist_squared / rpm / market / 31 | 4670 | 137.9375 | 623.1251 | 5.7181 | 3.2091 | 0.8327 |
| 18 | hist_absolute / rpm / market / 15 | 4670 | 116.1845 | 620.7098 | 4.6939 | 2.2960 | 0.8340 |
| 19 | hist_absolute / rpm / market / 31 | 4670 | 115.0497 | 619.9328 | 4.6385 | 2.2311 | 0.8344 |
| 20 | ridge / log_rate / market / 1 | 4670 | 113.5666 | 619.5003 | 4.6806 | 2.2908 | 0.8346 |
| 21 | ridge / log_rate / market / 100 | 4670 | 112.5039 | 618.9600 | 4.6202 | 2.2417 | 0.8349 |
| 22 | hist_squared / log_rate / market / 15 | 4670 | 122.8344 | 621.8660 | 5.0579 | 2.5691 | 0.8333 |
| 23 | hist_squared / log_rate / market / 31 | 4670 | 126.6144 | 622.0882 | 5.2341 | 2.7337 | 0.8332 |
| 24 | hist_absolute / log_rate / market / 15 | 4670 | 121.9451 | 622.1963 | 4.9528 | 2.4725 | 0.8332 |
| 25 | hist_absolute / log_rate / market / 31 | 4670 | 118.3196 | 620.1942 | 4.7949 | 2.3707 | 0.8342 |

## Frozen-Choice Holdout Results

Each model/target/signal family below uses its September-selected hyperparameter. Selected marks the final choice within each purpose. These tables are reporting only, with no October-driven changes.

| index | purpose | window | selected | candidate | rows | MAE | RMSE | MAPE_pct | MdAPE_pct | R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | validation | October (fit Jan-Sep) | False | median / rpm / none / 0 | 4853 | 253.8479 | 695.0469 | 11.8120 | 7.7774 | 0.7932 |
| 1 | validation | Sep-Oct (fit Jan-Aug) | False | median / rpm / none / 0 | 9523 | 256.9529 | 684.2479 | 11.6315 | 8.0319 | 0.7990 |
| 2 | validation | October (fit Jan-Sep) | False | equipment_median / rpm / none / 0 | 4853 | 231.3050 | 683.8653 | 10.9273 | 6.7421 | 0.7998 |
| 3 | validation | Sep-Oct (fit Jan-Aug) | False | equipment_median / rpm / none / 0 | 9523 | 229.0966 | 670.6600 | 10.4874 | 6.8014 | 0.8069 |
| 4 | validation | October (fit Jan-Sep) | False | ridge / rpm / none / 1 | 4853 | 143.5701 | 648.9020 | 6.6416 | 3.2887 | 0.8198 |
| 5 | validation | Sep-Oct (fit Jan-Aug) | False | ridge / rpm / none / 1 | 9523 | 146.4109 | 634.6188 | 6.4549 | 3.6220 | 0.8271 |
| 6 | validation | October (fit Jan-Sep) | False | hist_squared / rpm / none / 15 | 4853 | 186.7730 | 662.8205 | 8.7187 | 4.6414 | 0.8120 |
| 7 | validation | Sep-Oct (fit Jan-Aug) | False | hist_squared / rpm / none / 15 | 9523 | 121.7590 | 634.4763 | 5.3630 | 2.4605 | 0.8271 |
| 8 | validation | October (fit Jan-Sep) | True | hist_absolute / rpm / none / 31 | 4853 | 108.6078 | 646.3833 | 5.0366 | 1.6889 | 0.8212 |
| 9 | validation | Sep-Oct (fit Jan-Aug) | True | hist_absolute / rpm / none / 31 | 9523 | 107.0214 | 633.7018 | 4.6418 | 1.8732 | 0.8276 |
| 10 | validation | October (fit Jan-Sep) | False | ridge / log_rate / none / 1 | 4853 | 120.9567 | 646.2340 | 5.7094 | 2.3080 | 0.8213 |
| 11 | validation | Sep-Oct (fit Jan-Aug) | False | ridge / log_rate / none / 1 | 9523 | 123.1587 | 630.9953 | 5.4984 | 2.6205 | 0.8290 |
| 12 | validation | October (fit Jan-Sep) | False | hist_squared / log_rate / none / 15 | 4853 | 121.3277 | 646.9877 | 5.8342 | 2.2968 | 0.8208 |
| 13 | validation | Sep-Oct (fit Jan-Aug) | False | hist_squared / log_rate / none / 15 | 9523 | 117.2959 | 635.1910 | 5.1606 | 2.2770 | 0.8268 |
| 14 | validation | October (fit Jan-Sep) | False | hist_absolute / log_rate / none / 31 | 4853 | 108.1134 | 647.1084 | 5.1728 | 1.7723 | 0.8208 |
| 15 | validation | Sep-Oct (fit Jan-Aug) | False | hist_absolute / log_rate / none / 31 | 9523 | 109.8698 | 633.9928 | 4.8314 | 2.0123 | 0.8274 |
| 16 | validation | October (fit Jan-Sep) | False | ridge / rpm / market / 100 | 4853 | 119.9233 | 646.1013 | 5.6380 | 2.2425 | 0.8213 |
| 17 | validation | Sep-Oct (fit Jan-Aug) | False | ridge / rpm / market / 100 | 9523 | 107.3591 | 631.9094 | 4.7663 | 1.8935 | 0.8285 |
| 18 | validation | October (fit Jan-Sep) | False | hist_squared / rpm / market / 15 | 4853 | 207.3046 | 668.4990 | 9.3781 | 5.5268 | 0.8087 |
| 19 | validation | Sep-Oct (fit Jan-Aug) | False | hist_squared / rpm / market / 15 | 9523 | 121.1766 | 635.3209 | 5.3635 | 2.4511 | 0.8267 |
| 20 | validation | October (fit Jan-Sep) | False | hist_absolute / rpm / market / 31 | 4853 | 108.9132 | 645.8038 | 5.0668 | 1.6777 | 0.8215 |
| 21 | validation | Sep-Oct (fit Jan-Aug) | False | hist_absolute / rpm / market / 31 | 9523 | 115.4700 | 636.5554 | 4.9630 | 2.2221 | 0.8260 |
| 22 | validation | October (fit Jan-Sep) | False | ridge / log_rate / market / 100 | 4853 | 107.1522 | 645.8713 | 5.0815 | 1.7693 | 0.8215 |
| 23 | validation | Sep-Oct (fit Jan-Aug) | False | ridge / log_rate / market / 100 | 9523 | 109.8925 | 633.6840 | 4.8395 | 1.9907 | 0.8276 |
| 24 | validation | October (fit Jan-Sep) | False | hist_squared / log_rate / market / 15 | 4853 | 134.6949 | 648.9963 | 6.5510 | 2.9987 | 0.8197 |
| 25 | validation | Sep-Oct (fit Jan-Aug) | False | hist_squared / log_rate / market / 15 | 9523 | 123.3540 | 637.8597 | 5.4123 | 2.5442 | 0.8253 |
| 26 | validation | October (fit Jan-Sep) | False | hist_absolute / log_rate / market / 31 | 4853 | 109.9563 | 646.4472 | 5.2261 | 1.7914 | 0.8211 |
| 27 | validation | Sep-Oct (fit Jan-Aug) | False | hist_absolute / log_rate / market / 31 | 9523 | 118.3768 | 637.0970 | 5.1263 | 2.3217 | 0.8257 |
| 28 | december | October (fit Jan-Sep) | False | median / rpm / none / 0 | 4853 | 253.8479 | 695.0469 | 11.8120 | 7.7774 | 0.7932 |
| 29 | december | Sep-Oct (fit Jan-Aug) | False | median / rpm / none / 0 | 9523 | 256.9529 | 684.2479 | 11.6315 | 8.0319 | 0.7990 |
| 30 | december | October (fit Jan-Sep) | False | equipment_median / rpm / none / 0 | 4853 | 231.3050 | 683.8653 | 10.9273 | 6.7421 | 0.7998 |
| 31 | december | Sep-Oct (fit Jan-Aug) | False | equipment_median / rpm / none / 0 | 9523 | 229.0966 | 670.6600 | 10.4874 | 6.8014 | 0.8069 |
| 32 | december | October (fit Jan-Sep) | False | ridge / rpm / none / 100 | 4853 | 112.8127 | 647.5235 | 5.2854 | 1.9604 | 0.8205 |
| 33 | december | Sep-Oct (fit Jan-Aug) | False | ridge / rpm / none / 100 | 9523 | 108.9648 | 631.7153 | 4.8520 | 1.9744 | 0.8286 |
| 34 | december | October (fit Jan-Sep) | False | hist_squared / rpm / none / 15 | 4853 | 112.2597 | 647.0670 | 5.3302 | 1.8901 | 0.8208 |
| 35 | december | Sep-Oct (fit Jan-Aug) | False | hist_squared / rpm / none / 15 | 9523 | 113.1715 | 631.5032 | 5.0712 | 1.9939 | 0.8288 |
| 36 | december | October (fit Jan-Sep) | True | hist_absolute / rpm / none / 15 | 4853 | 112.9010 | 651.0015 | 5.1895 | 1.9353 | 0.8186 |
| 37 | december | Sep-Oct (fit Jan-Aug) | True | hist_absolute / rpm / none / 15 | 9523 | 108.8517 | 633.0133 | 4.7545 | 1.8938 | 0.8279 |
| 38 | december | October (fit Jan-Sep) | False | ridge / log_rate / none / 1 | 4853 | 110.6372 | 649.0912 | 5.1736 | 1.8896 | 0.8197 |
| 39 | december | Sep-Oct (fit Jan-Aug) | False | ridge / log_rate / none / 1 | 9523 | 107.3933 | 633.1965 | 4.7439 | 1.9228 | 0.8278 |
| 40 | december | October (fit Jan-Sep) | False | hist_squared / log_rate / none / 15 | 4853 | 113.6632 | 650.4816 | 5.3120 | 2.0173 | 0.8189 |
| 41 | december | Sep-Oct (fit Jan-Aug) | False | hist_squared / log_rate / none / 15 | 9523 | 109.1228 | 632.4591 | 4.8416 | 1.9600 | 0.8282 |
| 42 | december | October (fit Jan-Sep) | False | hist_absolute / log_rate / none / 15 | 4853 | 116.2917 | 651.4948 | 5.4037 | 2.1387 | 0.8183 |
| 43 | december | Sep-Oct (fit Jan-Aug) | False | hist_absolute / log_rate / none / 15 | 9523 | 109.6626 | 632.6743 | 4.8506 | 1.9963 | 0.8281 |

## Holdout Slices

Lanes are ordered pickup/delivery pairs and are marked seen only against the corresponding fitting window. Distance buckets use the supplied miles. Small unseen-lane slices are unstable estimates. All candidate slices are saved in outputs/holdout_slices.csv; the selected models are shown here.

| index | purpose | window | slice | group | rows | MAE | RMSE | MAPE_pct | MdAPE_pct | R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 80 | validation | October (fit Jan-Sep) | lane | seen | 4845 | 108.7460 | 646.9149 | 5.0432 | 1.6919 | 0.8209 |
| 81 | validation | October (fit Jan-Sep) | lane | unseen | 8 | 24.8748 | 38.2560 | 1.0716 | 0.8045 | 0.9993 |
| 82 | validation | October (fit Jan-Sep) | equipment | Dry Van | 2745 | 92.8812 | 620.0474 | 4.2775 | 1.3673 | 0.8154 |
| 83 | validation | October (fit Jan-Sep) | equipment | Reefer | 1190 | 124.4408 | 711.3256 | 5.8475 | 1.7177 | 0.8153 |
| 84 | validation | October (fit Jan-Sep) | equipment | Flatbed | 918 | 135.1090 | 635.0154 | 6.2553 | 3.3518 | 0.8327 |
| 85 | validation | October (fit Jan-Sep) | distance | 1500+ | 1364 | 193.2295 | 977.3358 | 5.0800 | 1.7126 | 0.3984 |
| 86 | validation | October (fit Jan-Sep) | distance | 1000-1500 | 890 | 121.8639 | 683.4041 | 4.8353 | 1.7892 | 0.1640 |
| 87 | validation | October (fit Jan-Sep) | distance | 500-1000 | 1520 | 71.5790 | 395.6069 | 4.7968 | 1.7089 | 0.3706 |
| 88 | validation | October (fit Jan-Sep) | distance | 0-250 | 290 | 25.8162 | 141.6731 | 5.4940 | 1.7714 | 0.4736 |
| 89 | validation | October (fit Jan-Sep) | distance | 250-500 | 789 | 49.1290 | 287.8871 | 5.4826 | 1.5222 | 0.2573 |
| 90 | validation | Sep-Oct (fit Jan-Aug) | lane | seen | 9502 | 107.1535 | 634.3950 | 4.6473 | 1.8728 | 0.8272 |
| 91 | validation | Sep-Oct (fit Jan-Aug) | lane | unseen | 21 | 47.2486 | 61.7005 | 2.1755 | 2.1310 | 0.9981 |
| 92 | validation | Sep-Oct (fit Jan-Aug) | equipment | Dry Van | 5360 | 101.7198 | 654.8354 | 4.2408 | 1.6379 | 0.8022 |
| 93 | validation | Sep-Oct (fit Jan-Aug) | equipment | Flatbed | 1770 | 110.7462 | 509.3428 | 5.3339 | 2.4509 | 0.8871 |
| 94 | validation | Sep-Oct (fit Jan-Aug) | equipment | Reefer | 2393 | 116.1412 | 667.6254 | 5.0282 | 2.0618 | 0.8309 |
| 95 | validation | Sep-Oct (fit Jan-Aug) | distance | 0-250 | 550 | 19.8651 | 103.5106 | 4.0897 | 2.0199 | 0.6129 |
| 96 | validation | Sep-Oct (fit Jan-Aug) | distance | 500-1000 | 2972 | 71.5919 | 367.8728 | 4.7742 | 1.8844 | 0.4069 |
| 97 | validation | Sep-Oct (fit Jan-Aug) | distance | 1000-1500 | 1760 | 123.3016 | 693.1424 | 4.4643 | 1.9018 | 0.1709 |
| 98 | validation | Sep-Oct (fit Jan-Aug) | distance | 1500+ | 2701 | 185.5629 | 954.0058 | 4.3969 | 1.8306 | 0.3962 |
| 99 | validation | Sep-Oct (fit Jan-Aug) | distance | 250-500 | 1540 | 50.1633 | 270.0321 | 5.2160 | 1.8350 | 0.2523 |
| 360 | december | October (fit Jan-Sep) | lane | seen | 4845 | 113.0243 | 651.5353 | 5.1947 | 1.9363 | 0.8183 |
| 361 | december | October (fit Jan-Sep) | lane | unseen | 8 | 38.2144 | 52.2225 | 2.0198 | 1.3728 | 0.9987 |
| 362 | december | October (fit Jan-Sep) | equipment | Dry Van | 2745 | 109.9640 | 623.3091 | 5.0136 | 2.0725 | 0.8134 |
| 363 | december | October (fit Jan-Sep) | equipment | Reefer | 1190 | 127.3543 | 719.2568 | 5.8953 | 1.9361 | 0.8112 |
| 364 | december | October (fit Jan-Sep) | equipment | Flatbed | 918 | 102.9475 | 638.8145 | 4.8005 | 1.6419 | 0.8307 |
| 365 | december | October (fit Jan-Sep) | distance | 1500+ | 1364 | 198.5052 | 985.1226 | 5.1301 | 1.8880 | 0.3888 |
| 366 | december | October (fit Jan-Sep) | distance | 1000-1500 | 890 | 127.8850 | 688.0991 | 5.0092 | 2.0068 | 0.1524 |
| 367 | december | October (fit Jan-Sep) | distance | 500-1000 | 1520 | 74.7189 | 396.8817 | 4.9156 | 1.9305 | 0.3666 |
| 368 | december | October (fit Jan-Sep) | distance | 0-250 | 290 | 27.4740 | 141.7042 | 5.8119 | 2.1645 | 0.4734 |
| 369 | december | October (fit Jan-Sep) | distance | 250-500 | 789 | 52.9654 | 290.0016 | 5.7944 | 1.9016 | 0.2464 |
| 370 | december | Sep-Oct (fit Jan-Aug) | lane | seen | 9502 | 108.9559 | 633.7006 | 4.7593 | 1.8923 | 0.8276 |
| 371 | december | Sep-Oct (fit Jan-Aug) | lane | unseen | 21 | 61.7066 | 82.1853 | 2.5891 | 2.3898 | 0.9967 |
| 372 | december | Sep-Oct (fit Jan-Aug) | equipment | Dry Van | 5360 | 111.8526 | 654.2778 | 4.6737 | 1.9228 | 0.8026 |
| 373 | december | Sep-Oct (fit Jan-Aug) | equipment | Flatbed | 1770 | 95.4059 | 506.5843 | 4.7981 | 1.8920 | 0.8884 |
| 374 | december | Sep-Oct (fit Jan-Aug) | equipment | Reefer | 2393 | 112.0755 | 667.8031 | 4.9030 | 1.8395 | 0.8308 |
| 375 | december | Sep-Oct (fit Jan-Aug) | distance | 0-250 | 550 | 19.9319 | 103.2659 | 4.1592 | 2.0582 | 0.6148 |
| 376 | december | Sep-Oct (fit Jan-Aug) | distance | 500-1000 | 2972 | 72.0250 | 367.6174 | 4.8535 | 1.8801 | 0.4077 |
| 377 | december | Sep-Oct (fit Jan-Aug) | distance | 1000-1500 | 1760 | 125.6086 | 691.3809 | 4.6198 | 1.9390 | 0.1751 |
| 378 | december | Sep-Oct (fit Jan-Aug) | distance | 1500+ | 2701 | 189.6643 | 953.3695 | 4.5315 | 1.8689 | 0.3970 |
| 379 | december | Sep-Oct (fit Jan-Aug) | distance | 250-500 | 1540 | 50.7921 | 269.8538 | 5.3210 | 1.8759 | 0.2533 |

## Separate December Model

The December model starts from pickup, delivery, distance, equipment, weight and date only. Coordinates are looked up exclusively from its fitting-window city map, including during September and October tests. Unmapped coordinates use fitting-window numeric medians. It uses weekday and day of month, with no month, week of year, market or quote. Both target choices and both histogram losses are compared again under this exact input restriction.

| index | candidate | rows | MAE | RMSE | MAPE_pct | MdAPE_pct | R2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | median / rpm / none / 0 | 4670 | 260.8321 | 673.3033 | 11.4477 | 8.3686 | 0.8046 |
| 1 | equipment_median / rpm / none / 0 | 4670 | 227.3228 | 657.0658 | 10.0364 | 6.9387 | 0.8139 |
| 2 | ridge / rpm / none / 1 | 4670 | 105.5616 | 614.9322 | 4.4176 | 2.0219 | 0.8370 |
| 3 | ridge / rpm / none / 100 | 4670 | 105.3190 | 614.7701 | 4.4209 | 2.0164 | 0.8371 |
| 4 | hist_squared / rpm / none / 15 | 4670 | 112.9874 | 614.4603 | 4.7407 | 2.0791 | 0.8373 |
| 5 | hist_squared / rpm / none / 31 | 4670 | 123.8635 | 616.9136 | 5.2585 | 2.3883 | 0.8360 |
| 6 | hist_absolute / rpm / none / 15 | 4670 | 100.5725 | 613.1188 | 4.1446 | 1.7194 | 0.8380 |
| 7 | hist_absolute / rpm / none / 31 | 4670 | 110.4215 | 613.2227 | 4.5385 | 2.0728 | 0.8379 |
| 8 | ridge / log_rate / none / 1 | 4670 | 103.6780 | 616.0401 | 4.2875 | 1.9291 | 0.8365 |
| 9 | ridge / log_rate / none / 100 | 4670 | 104.9320 | 616.3743 | 4.3133 | 1.9694 | 0.8363 |
| 10 | hist_squared / log_rate / none / 15 | 4670 | 103.8366 | 613.7005 | 4.3259 | 1.8786 | 0.8377 |
| 11 | hist_squared / log_rate / none / 31 | 4670 | 110.6603 | 613.5758 | 4.6183 | 2.0779 | 0.8378 |
| 12 | hist_absolute / log_rate / none / 15 | 4670 | 102.8609 | 613.3884 | 4.2919 | 1.8582 | 0.8379 |
| 13 | hist_absolute / log_rate / none / 31 | 4670 | 105.1314 | 613.1163 | 4.3989 | 1.9904 | 0.8380 |

Selected December configuration: {'model': 'hist_absolute', 'target': 'rpm', 'market': False, 'december': True, 'parameter': 15}. Its final chart spans 814.1127 to 846.8442, a range of 3.9375% of the mean. Holding each row's other features fixed and changing only its weekday gives a maximum rate spread of 38.107392. This explicitly tests whether the fitted model uses weekday.

The chart is nearly flat relative to its mean rate. The scorer uses a narrow vertical axis, which makes the small calendar changes look larger. These changes should not be described as a daily market forecast.

## Final Prediction Checks

Refit both choices on all 48,000 training rows. Wrote 12,000 validation predictions in template order and 31 December rows with their original columns. There are 0 missing validation model-feature cells, 0 missing validation output cells and 0 missing December output cells. Positive finite predictions are asserted before writing.

| index | train_rpm | predicted_rpm |
| --- | --- | --- |
| count | 48,000 | 12,000 |
| mean | 2.2153 | 2.1962 |
| std | 0.5839 | 0.2631 |
| min | 0.3337 | 1.6512 |
| 1% | 1.6735 | 1.7729 |
| 5% | 1.8075 | 1.8460 |
| 50% | 2.1453 | 2.1573 |
| 95% | 2.7359 | 2.6988 |
| 99% | 3.1769 | 2.9870 |
| max | 14.1253 | 3.3900 |

## Distance Policy

An earlier version replaced the supplied distance when it was more than three times the air miles, using 1.18 times the air miles. The data does not support that. The 12 training rows above that ratio are all New Orleans to Shreveport, Shreveport to New Orleans at a distance of 70 miles, and 70 is also the smallest distance in the whole training file (272 loads are 100 miles or shorter, so it looks like a distance floor). These loads sold for 197 to 232 (median 216), which is 3.08 per mile on the supplied distance, in line with the 2.95 per mile of all short loads. Using the repaired distance would have missed those rates by a median of 88 percent. So the distance is right and the city coordinates are the approximate part (two cities only 7 miles apart on the map). The supplied distance is now used everywhere; the ratio stays a feature and a flag.

Flags cover 12 training rows and 11 validation rows. Their predictions are in line with the rest of the book.

| index | load_id | pickup | delivery | distance | predicted_rate | predicted_rpm |
| --- | --- | --- | --- | --- | --- | --- |
| 1433 | TE-001434 | New Orleans | Shreveport | 70 | 216.6880 | 3.0955 |
| 2666 | TE-002667 | New Orleans | Shreveport | 70 | 237.3004 | 3.3900 |
| 2907 | TE-002908 | New York | Allentown | 70 | 191.1908 | 2.7313 |
| 4111 | TE-004112 | Shreveport | New Orleans | 70 | 215.2401 | 3.0749 |
| 6058 | TE-006059 | New York | Allentown | 70 | 197.4741 | 2.8211 |
| 6505 | TE-006506 | Allentown | New York | 70 | 192.9590 | 2.7566 |
| 7828 | TE-007829 | Shreveport | New Orleans | 70 | 218.0444 | 3.1149 |
| 9074 | TE-009075 | Shreveport | New Orleans | 70 | 213.4001 | 3.0486 |
| 9704 | TE-009705 | New Orleans | Shreveport | 70 | 216.6342 | 3.0948 |
| 9743 | TE-009744 | Shreveport | New Orleans | 70 | 232.5563 | 3.3222 |
| 10497 | TE-010498 | New Orleans | Shreveport | 70 | 211.5838 | 3.0226 |

As an empirical plausibility screen, the training RPM 0.1% and 99.9% quantiles are 0.4374 and 10.0688. There are 0 validation predictions outside this band. This is a data-based screen, not an external freight-pricing guarantee. No prices were clipped to this band.

Validation includes 1,461 rows on unseen directed lanes. Coordinate features permit predictions there, but the small temporal unseen-lane slices do not establish reliable performance for every new city.

## Reproduction

Run from the worktree with the supplied Python environment. Raw CSVs remain outside the repo. Both scripts default to --data-dir data. The scorer checks file validity and generates the chart; it does not report hidden-label accuracy.

```powershell
$python = 'D:\Assessments\Spotter\freight-rate\.venv\Scripts\python.exe'
$data = 'D:\Assessments\Spotter\freight-rate\data'
$env:OMP_NUM_THREADS = '1'
& $python src/quote_regimes.py --data-dir $data
& $python src/train.py --data-dir $data
& $python score.py --predictions validation_predictions.csv --december-predictions outputs/december_chart_inputs_predicted.csv
```

## Recommendation

Use the absolute-error HistGradientBoostingRegressor on RPM with no signals and 31 maximum leaf nodes (hist_absolute / rpm / none / 31). October MAE is 108.61, RMSE 646.38, MAPE 5.04%, median APE 1.69% and R2 0.8212. The two-month January-August-fit check has MAE 107.02, RMSE 633.70, MAPE 4.64%, median APE 1.87% and R2 0.8276. The much larger RMSE indicates that large errors remain despite the low typical percentage error.

For a two to three minute review, explain the time split and September-only selection first. Show the direct, mirrored and unresolved quote relationships and why quote is excluded. Explain the distance policy, fitting-window weight imputation and coordinate features for unseen cities. Compare the selected model with the median baselines, then show the separate December result. Name the remaining risks plainly: future quote regimes are unknown, unseen-lane evidence is limited, and a nearly flat chart cannot represent unobserved daily market movements. Any use of market_index also depends on confirming that it is available before the rate is quoted.
