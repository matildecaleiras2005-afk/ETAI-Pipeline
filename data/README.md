# Dataset -- COMPAS Recidivism (ProPublica)

## The problem

In 2016, ProPublica investigated COMPAS, a risk-assessment algorithm
actually used by courts in Broward County, Florida, to help inform
bail and sentencing decisions. COMPAS scores a defendant's likelihood
of reoffending on a 1-10 scale; judges could see that score when
deciding, among other things, whether someone should be released
before trial. ProPublica obtained COMPAS's scores for thousands of
defendants and matched them against what actually happened over the
following two years, then published the data.

This dataset is that data: each row is one defendant, with their
demographics and criminal history at the time of screening, COMPAS's
own risk score for them, and whether they were actually rearrested
within two years.

**Your task:** predict `two_year_recid` -- will this person be
rearrested within two years? -- from the case facts. Once you have a
model, the more interesting question is the one ProPublica actually
asked: is it equally accurate for everyone, or does it get things
wrong more often, in a particular direction, for some groups than
others? `race` is deliberately excluded from the model's own inputs
(see `config.yaml` and `src/preprocessing.py`) so it can be used
afterward purely to check this, in `src/evaluate.py`.

Before any of that: look at the data first. It comes from a real
system with real data-entry and record-keeping quirks -- don't assume
every column is clean or consistent just because it loads without
error.

## Data dictionary

| column | type | description | notable values |
|--------|------|--------------|------------------|
| `id` | identifier | internal record id | not a model feature |
| `sex` | categorical | defendant's sex | `Male`, `Female` |
| `age` | numeric | defendant's age (years) at screening | |
| `age_cat` | categorical | age bucket | `Less than 25`, `25 - 45`, `Greater than 45` |
| `race` | categorical | defendant's race, as recorded | `African-American`, `Caucasian`, `Hispanic`, `Asian`, `Native American`, `Other`; excluded from model features, used only to audit fairness |
| `juv_fel_count` | numeric | number of prior juvenile felony offenses | |
| `juv_misd_count` | numeric | number of prior juvenile misdemeanor offenses | |
| `juv_other_count` | numeric | number of other prior juvenile offenses | |
| `juvenile_total` | numeric | total juvenile offenses | |
| `priors_count` | numeric | number of prior adult offenses | |
| `prior_offenses` | numeric | number of prior offenses | |
| `age_in_months` | numeric | age expressed in months | |
| `c_charge_degree` | categorical | degree of the current charge | `F` (felony), `M` (misdemeanor) |
| `decile_score` | numeric | COMPAS's own risk score | 1 (lowest risk) to 10 (highest risk); excluded from model features, used only for comparison |
| `score_text` | categorical | COMPAS's own risk category | `Low`, `Medium`, `High`; excluded from model features, used only for comparison |
| `two_year_recid` | binary | **target** -- was this person rearrested within two years? | `0` = no, `1` = yes |

Source: derived from [propublica/compas-analysis](https://github.com/propublica/compas-analysis) (the data behind the "Machine Bias" investigation). Personally-identifying columns (name, date of birth, case numbers, charge descriptions) were removed.

20260671 Matilde Rodrigues Caleiras 

-> CLASS - WEEK 2

Run: 20260916_120811
Model: logistic_regression  params={'max_iter': 1000}
Test size: 0.2  random_state: 42
============================================================

Train accuracy: 0.678
Test accuracy:  0.678
Gap (train - test): -0.000


Run: 20260916_122724
Model: decision_tree  params={'max_depth': 5}
Test size: 0.2  random_state: 42
============================================================

Train accuracy: 0.680
Test accuracy:  0.668
Gap (train - test): +0.012

Conclusions: 

Logistic Regression outperforms the Decision Tree with higher test accuracy (0.678 vs 0.668).
Generalization is perfect for Logistic Regression (gap: 0.000), whereas the Decision Tree shows minor overfitting (gap: +0.012).
So, in conclusion logistic regression is the better model due to superior accuracy and stability.

-> CLASS - WEEK 3

Run: 20260923_173415
Model: logistic_regression  params={'max_iter': 1000}
Test size: 0.2  random_state: 42
============================================================

Train accuracy: 0.678
Test accuracy:  0.655
Gap (train - test): +0.023


Run: 20260923_173748
Model: decision_tree  params={'max_depth': 5}
Test size: 0.2  random_state: 42
============================================================

Train accuracy: 0.691
Test accuracy:  0.642
Gap (train - test): +0.049

Conclusions: 

Model Comparison (Post-Cleaning):
   - Logistic Regression outperformed the Decision Tree across all metrics post-cleaning, achieving higher test accuracy (0.655 vs. 0.642) and demonstrating superior generalization capability.
   - Logistic Regression showed better stability with a lower generalization gap (+0.023), whereas the Decision Tree displayed a higher discrepancy between train and test metrics (+0.049), indicating a mild tendency toward overfitting, likely due to tree depth complexity (max_depth=5).
   - So, in conclusion, Logistic Regression provides a more robust, stable, and interpretable baseline model compared to Decision Tree constrained at max_depth=5.

Impact of Data Cleaning (Week 2 vs. Week 3):
   - Test accuracy dropped slightly across both models (Logistic Regression: 0.678 to  0.655; and Decision Tree: 0.668 to 0.642). This performance adjustment suggests that the cleaning pipeline effectively removed noisy instances or redundant observations from the raw dataset. 
   - The Decision Tree showed higher sensitivity to the pipeline updates, with its train-test gap increasing from +0.012 to +0.049. In contrast, Logistic Regression maintained a smaller generalization gap (+0.023), reaffirming its superior robustness and consistency as a baseline classifier for this dataset.