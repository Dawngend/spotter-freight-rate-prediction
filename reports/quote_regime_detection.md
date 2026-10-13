# Quote Signal Regime Detection

The monthly regime of quote_signal is read from columns that also exist in validation, without posted_rate. In a direct month quote_signal follows rate per mile, which falls as distance rises, so quote_signal and distance are negatively correlated. In a mirrored month the relation flips and the correlation is positive. In an unresolved month (August) quote_signal is noise and the correlation is near zero. The statistic is the Spearman correlation of quote_signal with distance inside each equipment type, averaged over the three types.

The threshold is half of the smallest absolute statistic among the resolved months from January to August, which is 0.440. It was fixed on January to August only and applied unchanged to September and October.

## Development Months (true regime shown only to check the detector)

| index | month | rows | statistic | true_regime | detected | correct |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 2025-01 | 4918 | -0.9031 | direct | direct | yes |
| 1 | 2025-02 | 4337 | -0.8920 | direct | direct | yes |
| 2 | 2025-03 | 5036 | -0.8796 | direct | direct | yes |
| 3 | 2025-04 | 4819 | 0.8835 | mirror | mirror | yes |
| 4 | 2025-05 | 4913 | 0.8913 | mirror | mirror | yes |
| 5 | 2025-06 | 4783 | -0.8906 | direct | direct | yes |
| 6 | 2025-07 | 4912 | 0.8804 | mirror | mirror | yes |
| 7 | 2025-08 | 4759 | 0.0199 | unresolved | unresolved | yes |
| 8 | 2025-09 | 4670 | -0.8946 | direct | direct | yes |
| 9 | 2025-10 | 4853 | 0.8924 | mirror | mirror | yes |

Detection accuracy: 100 percent over January to October, and 100 percent on the two held-out months (September and October).

## Validation Months (no labels)

| index | month | rows | statistic | largest_weekly_abs_correlation | quote_mean | quote_std | detected |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2025-11 | 5836 | 0.0186 | 0.0500 | 2.0553 | 0.2203 | unresolved |
| 1 | 2025-12 | 6164 | -0.0008 | 0.0340 | 2.0476 | 0.2193 | unresolved |

For comparison, August (the unresolved month) has quote_signal mean 2.052, standard deviation 0.223, minimum 1.260 and maximum 2.881.

## Conclusion

Both validation months are unresolved: the statistic is near zero and no week reaches the size of a resolved month, and the quote_signal distribution is August-like. In other words quote_signal is noise in November and December. A regime-aware model has nothing to use, so no alternative prediction file is written and quote_signal stays excluded from the final model. This confirms the earlier decision with a direct check instead of a classifier.
