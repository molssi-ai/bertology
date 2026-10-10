# 01-ft-random-cv-methods

**Which of the eight methods differ, on the random split?**

Design. 8 methods x 5 paired folds per endpoint, one KFold split at seed 1234, bin 5. The fold is the subject.

These results should be considered as supporting evidence, not inferential.

Five samples is below Dietterich's 5x2 and far below the 25 that guidelines in
Ash et al. ask for, and the variance-homogeneity assumption cannot be cleared at
this sample size (see assumptions_all.csv). Both the parametric and the
non-parametric route are shipped here so the reader can see they agree or do
not.

## What came out

```
endpoint  metric  n  k      anova_p  anova_significant      best  best_mean  n_pairs  n_significant  n_indistinguishable_from_best  var_ratio  var_ratio_ok
     HLM     mae  5  8 4.504821e-10               True BERT-Base   0.341467       28             16                              4   4.003747          True
     HLM    rmse  5  8 2.386309e-11               True BERT-Base   0.440710       28             14                              3   5.299100          True
     HLM pearson  5  8 1.555354e-11               True BERT-Base   0.708078       28             14                              4   8.789299          True
     HLM      r2  5  8 1.558210e-11               True BERT-Base   0.498397       28             13                              4   9.525970         False
 MDR1_ER     mae  5  8 1.690241e-08               True BERT-Base   0.335748       28             14                              3   7.760509          True
 MDR1_ER    rmse  5  8 2.331586e-09               True       SVM   0.449048       28             13                              2   3.783769          True
 MDR1_ER pearson  5  8 4.926931e-09               True       SVM   0.756557       28             10                              5   3.628558          True
 MDR1_ER      r2  5  8 1.357189e-09               True       SVM   0.569530       28             12                              3   4.093463          True
     RLM     mae  5  8 1.694800e-12               True BERT-Base   0.399256       28             17                              1  17.624377         False
     RLM    rmse  5  8 1.126108e-14               True BERT-Base   0.510762       28             18                              2   5.146602          True
     RLM pearson  5  8 2.323121e-15               True BERT-Base   0.735478       28             17                              2   5.963023          True
     RLM      r2  5  8 8.546298e-15               True BERT-Base   0.533997       28             18                              2   5.894970          True
     SOL     mae  5  8 9.888567e-05               True BERT-Base   0.366055       28              6                              2   8.014741          True
     SOL    rmse  5  8 3.930693e-06               True BERT-Base   0.543861       28              9                              4   2.959738          True
     SOL pearson  5  8 1.041226e-06               True BERT-Base   0.615314       28             10                              3  14.671842         False
     SOL      r2  5  8 4.192235e-06               True BERT-Base   0.361548       28              9                              4   9.371197         False
    hPPB     mae  5  8 3.612166e-13               True       SVM   0.350022       28             14                              2  31.411964         False
    hPPB    rmse  5  8 1.623977e-12               True       SVM   0.462881       28             12                              3   6.060273          True
    hPPB pearson  5  8 7.245749e-14               True       SVM   0.769804       28             10                              5   3.975025          True
    hPPB      r2  5  8 5.820155e-13               True       SVM   0.589135       28             11                              4   7.712658          True
    rPPB     mae  5  8 3.520676e-08               True       SVM   0.391569       28             12                              3   5.163370          True
    rPPB    rmse  5  8 3.239419e-06               True       SVM   0.529565       28              7                              6   8.829363          True
    rPPB pearson  5  8 1.313642e-07               True       SVM   0.726758       28              7                              6   6.087903          True
    rPPB      r2  5  8 1.582927e-06               True       SVM   0.515373       28              7                              6   6.397535          True
```


## How the numbers were produced

Parametric route, which is what the Ash et al.'s protocol recommends:
repeated-measures ANOVA, then Tukey HSD computed on the repeated-measures error
term rather than a between-subject one, so the pairing is used. The significance
level is Bonferroni divided across the metrics at the ANOVA step; Tukey controls
the family-wise error across all pairwise comparisons inside a metric.

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


(Abbr.) EP: Endpoint