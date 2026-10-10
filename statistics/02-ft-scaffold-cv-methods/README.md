# Finetuning of MPP Models on ADME Dataset Using 5-Fold CV and Scaffold Splits

**Which of the eight methods differ, on the scaffold split?**

Design. 8 methods x 15 paired samples per endpoint, 3 split seeds x 5 folds, bin 5.
The subject is the (seed, fold) pair.

These results are the closest available approximation to Ash et al.'s guidelines
and the only fine-tuning stage whose parametric assumptions are satisfied in
every cell. Fifteen samples is still below the recommended 25, so the tests are
less powerful than the protocol intends.

## What came out

```
endpoint  metric  n  k      anova_p  anova_significant      best  best_mean  n_pairs  n_significant  n_indistinguishable_from_best  var_ratio  var_ratio_ok
     HLM     mae 15  8 4.941433e-43               True BERT-Base   0.342074       28             21                              0   1.485902          True
     HLM    rmse 15  8 3.664578e-42               True BERT-Base   0.452877       28             22                              0   2.006922          True
     HLM pearson 15  8 9.282776e-43               True BERT-Base   0.692068       28             21                              0   3.910298          True
     HLM      r2 15  8 1.163211e-43               True BERT-Base   0.471058       28             22                              0   2.091839          True
 MDR1_ER     mae 15  8 1.319126e-36               True BERT-Base   0.337280       28             24                              1   4.755941          True
 MDR1_ER    rmse 15  8 3.924833e-40               True       SVM   0.453990       28             23                              1   2.201123          True
 MDR1_ER pearson 15  8 4.033078e-37               True       SVM   0.752387       28             21                              2   5.568372          True
 MDR1_ER      r2 15  8 5.012184e-40               True       SVM   0.562449       28             23                              1   4.123364          True
     RLM     mae 15  8 4.053515e-38               True BERT-Base   0.400000       28             21                              0   3.258197          True
     RLM    rmse 15  8 3.417067e-40               True BERT-Base   0.513564       28             21                              0   2.975797          True
     RLM pearson 15  8 1.199052e-36               True BERT-Base   0.732692       28             19                              1   5.568606          True
     RLM      r2 15  8 3.147869e-40               True BERT-Base   0.532139       28             20                              1   3.415262          True
     SOL     mae 15  8 7.193147e-18               True BERT-Base   0.372847       28             13                              1   3.486465          True
     SOL    rmse 15  8 1.204462e-19               True BERT-Base   0.548904       28             20                              1   1.667083          True
     SOL pearson 15  8 1.310535e-23               True BERT-Base   0.608646       28             20                              1   3.092357          True
     SOL      r2 15  8 2.584026e-20               True BERT-Base   0.352283       28             20                              1   3.017446          True
    hPPB     mae 15  8 9.913961e-33               True       SVM   0.384945       28             19                              0   1.720858          True
    hPPB    rmse 15  8 2.598484e-29               True       SVM   0.507341       28             16                              2   2.736183          True
    hPPB pearson 15  8 2.334603e-33               True       SVM   0.714853       28             15                              2   1.901206          True
    hPPB      r2 15  8 1.580687e-30               True       SVM   0.503714       28             17                              2   1.772608          True
    rPPB     mae 15  8 1.687478e-17               True       SVM   0.420936       28             13                              3   1.847723          True
    rPPB    rmse 15  8 3.048008e-14               True       SVM   0.550213       28             11                              3   1.615616          True
    rPPB pearson 15  8 1.330362e-15               True       SVM   0.686433       28              9                              4   3.705298          True
    rPPB      r2 15  8 1.533224e-14               True       SVM   0.459489       28             11                              3   2.121289          True
```


## How the numbers were produced

Parametric route, which is what the protocol recommends: repeated-measures
ANOVA, then Tukey HSD computed on the repeated-measures error term rather than a
between-subject one, so the pairing is used. The significance level is
Bonferroni divided across the metrics at the ANOVA step; Tukey controls the
family-wise error across all pairwise comparisons inside a metric.

Non-parametric route, always computed and shipped beside it: Friedman with the
Conover-Friedman post hoc and Holm-Bonferroni. It is the documented fallback, and
it is written out whether or not the assumptions favour it, so the choice is
visible rather than hidden.

Assumptions are reviewed in the order the Supporting Information ranks them.
Independence is argued from the design, not tested. Homogeneity of variance is
checked against the ratio rule, at most 3 and at most 9 for equally sized groups.
Normality is judged from the residual quantile-quantile plots under
`results/figures/diagnostics/`. Shapiro-Wilk is recorded in `assumptions_all.csv`
and flagged as not decisive, because the Supporting Information warns those tests
have low power here and are more sensitive to their own assumptions than the ANOVA
is to non-normality.

Every p-value is accompanied by an effect size: the mean difference with its Tukey
interval, and Cohen's d in the pooled-variance form. For MAE and RMSE the
difference is also given as a fold error, since the endpoints are on a log scale.

## Files

    results/tables/anova_all.csv                        one row per endpoint and metric
    results/tables/tukey_<EP>_<metric>.csv              every pair: difference, interval, adjusted p, d
    results/tables/friedman_conover_<EP>_<metric>.csv   the non-parametric route
    results/tables/assumptions_all.csv                  variance ratio, Shapiro, Greenhouse-Geisser
    results/tables/cld_<EP>.csv / .tex                  compact letter display; shared letter means no significant difference
    results/tables/summary.csv                          one line per endpoint and metric
    results/figures/simultaneous_ci_<EP>                intervals adjusted so that non-overlap implies significance
    results/figures/mcsim_<EP>                          effect size as colour, significance as stars
    results/figures/forest_<EP>                         every pairwise difference against zero
    results/figures/diagnostics/                        residual QQ plots
