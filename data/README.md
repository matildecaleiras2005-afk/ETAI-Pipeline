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

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

-> CLASS - WEEK 4

Run: 20260930_153547
Model: logistic_regression  params={'max_iter': 1000}
Test size: 0.2  random_state: 42
============================================================

Train accuracy: 0.678
Test accuracy:  0.674
Gap (train - test): +0.004

Run: 20260930_153815
Model: decision_tree  params={'max_depth': 5}
Test size: 0.2  random_state: 42
============================================================

Train accuracy: 0.685
Test accuracy:  0.678
Gap (train - test): +0.006

Run: 20260930_154304
Model: random_forest  params={'n_estimators': 300, 'random_state': 42, 'n_jobs': -1}
Test size: 0.2  random_state: 42
============================================================

Train accuracy: 0.721
Test accuracy:  0.644
Gap (train - test): +0.078

Run: 20260930_154827
Model: dummy  params={'strategy': 'most_frequent'}
Test size: 0.2  random_state: 42
============================================================

Train accuracy: 0.549
Test accuracy:  0.550
Gap (train - test): -0.000

Conclusions: 

Model Comparison (Post-Preprocessing):
   - The dummy baseline yields a 55% validation accuracy by predicting the majority class, which all parameterized models successfully outperform, proving effective feature learning post-preprocessing.
   - Logistic Regression (67.4%) and Decision Tree (67.8%) achieve the highest validation performance, both demonstrating strong generalization with minimal train-validation gaps (+0.004 and +0.006) and no overfitting.
   - Random Forest scores the highest train accuracy (72.1%) but drops to 64.4% on validation, where an unconstrained depth causes a +0.078 gap due to memorizing training noise.
   - This comparison highlights the trade-off between top validation score and model robustness; while Decision Tree achieves marginally higher accuracy (67.8%), Logistic Regression (67.4%) is, for me, selected as the overall preferred model due to its superior stability (+0.004 gap) and lower risk of overfitting.

Impact of Preprocessing (Week 3 vs. Week 4):
   - Comparing Week 3 (data cleaning only) to Week 4 (with preprocessing), we can observe a clear performance gain across both evaluated models (Logistic Regression and Decision Tree).
   - About the Logistic Regression, adding scaling, encoding, and imputation we managed to increase test accuracy from 65.5% to 67.4% (+1.9%) while drastically reducing the train-test gap from +0.023 to +0.004, confirming that linear models heavily benefit from structured feature preprocessing.
   - The Decision Tree showed an even larger gain in test accuracy, rising from 64.2% to 67.8% (+3.6%), while its generalization gap dropped from +0.049 to +0.006, proving that robust preprocessing effectively mitigated its earlier overfitting.
   - So, in conclusion, while Logistic Regression was the clear superior model post-cleaning in Week 3, the Week 4 preprocessing pipeline elevated both models to comparable top-tier performance (~67.4%-67.8%), eliminating previous instability and validating the pipeline's impact on generalization.

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

-> CLASS - WEEK 5 (last week that was not evaluated)

| Model | Default: CV mean ± std | Tuned: nested CV mean ± std | Tuning score (best trial) | Optimism | Chosen hyperparameters |
| --- | --- | --- | --- | --- | --- |
| Logistic regression | 0.672 ± 0.013 | 0.673 ± 0.016 | 0.678 | +0.002 | {'model__C': 0.0715}` |
| Decision tree | 0.674 ± 0.018 | 0.672 ± 0.017 | 0.680 | +0.002 | {'model__criterion': 'gini', 'model__max_depth': 9, 'model__min_samples_leaf': 95} |
| Random forest | 0.650 ± 0.018 | 0.681 ± 0.015 | 0.683 | +0.000 | {'model__max_depth': 29, 'model__max_features': 'sqrt', 'model__min_samples_leaf': 63} |

Conclusions: 

- Hyperparameter tuning provided significant performance gains for Random Forest, raising its honest nested CV validation accuracy from 0.650 to 0.681 (+3.1 percentage points) and eliminating its overfitting gap (from +0.083 to +0.008). In contrast, Logistic Regression (0.672 to 0.673) and Decision Tree (0.674 to 0.672) showed minimal changes that were smaller than the fold-to-fold standard deviations (between 0.013 - 0.017).

- So, tuning successfully altered model selection, elevating Random Forest to become the best model with an honest validation score of 0.681 (compared to 0.672 - 0.673 for the linear and simple tree baselines). I decided to report the honest nested CV score of 0.681 rather than the optimistic best trial score of 0.683 to ensure an unbiased estimate of generalization performance on unseen test data.

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

-> CLASS - WEEK 6 

PART A: Week 5 Results + New Models' Results (Oblique RF and AdaBoost)

| Model | Default: CV mean ± std (gap) | Tuned: nested CV mean ± std (gap) | Tuning score (best trial) | Chosen hyperparameters |
|---|---|---|---|---|
| Logistic regression | 0.672 ± 0.013 (+0.003) | 0.673 ± 0.016 (+0.004) | 0.678 | {'model__C': 0.0715} |
| Decision tree | 0.674 ± 0.018 (+0.011) | 0.672 ± 0.017 (+0.013) | 0.680 | {'model__criterion': 'gini', 'model__max_depth': 9, 'model__min_samples_leaf': 95} |
| Random forest | 0.650 ± 0.018 (+0.083) | 0.681 ± 0.015 (+0.008) | 0.683 | {'model__max_depth': 29, 'model__max_features': 'sqrt', 'model__min_samples_leaf': 63} |
| Oblique random forest | 0.648 ± 0.012 (+0.114) | 0.680 ± 0.011 (+0.008) | 0.684 | {'model__estimator__project__n_components': 5, 'model__estimator__tree__max_depth': 6, 'model__estimator__tree__min_samples_leaf': 3} |
| AdaBoost | 0.669 ± 0.013 (+0.006) | 0.682 ± 0.017 (+0.004) | 0.684 | {'model__estimator__max_depth': 4, 'model__learning_rate': 0.025, 'model__n_estimators': 220} |

Conclusions:

- I would choose AdaBoost as my primary model, as it achieved the highest honest nested CV validation score of 0.682 (followed by Random Forest at 0.681 and Oblique RF at 0.680), though its lead over simpler models like Logistic Regression and Decision Tree (0.672–0.673) is small and well within the fold-to-fold standard deviations (~0.011 - 0.017).

- The chosen hyperparameters align with each model's theoretical framework: AdaBoost selected relatively shallow trees (max_depth: 4) to combine weak learners sequentially without overfitting, while Random Forest utilized deeper trees (max_depth: 29) constrained by a large leaf size (min_samples_leaf: 63) to reduce variance through bagging.

- As in week 5, for the selected model (AdaBoost), I prefer using the honest nested CV score of 0.682 rather than the optimistic whole-dataset tuning score of 0.684, as it gives a much more reliable picture of its true performance on new data.

PART B: Challenge 1 Results 

Note: The engineered features (high_priors_flag and weighted_criminal_load) are computed deterministically prior to data splitting, entirely preventing any form of data leakage.

| Model | Default: CV mean ± std | Tuned: nested CV mean ± std | Tuning score (best trial) | Optimism | Chosen hyperparameters |
|---|---|---|---|---|---|
| Random Forest (Without FE) | 0.650 ± 0.018 | 0.681 ± 0.015 | 0.683 | +0.000 | {'model__max_depth': 29, 'model__min_samples_leaf': 63, 'model__max_features': 'sqrt'} |
| Random Forest (With FE) | 0.646 ± 0.017 | 0.681 ± 0.013 | 0.682 | +0.000 | {'model__max_depth': 26, 'model__min_samples_leaf': 28, 'model__max_features': 'sqrt'} |

| Model | Default: CV mean ± std | Tuned: nested CV mean ± std | Tuning score (best trial) | Optimism | Chosen hyperparameters |
|---|---|---|---|---|---|
| AdaBoost (Without FE) | 0.669 ± 0.013 | 0.682 ± 0.017 | 0.684 | -0.000 | {'model_estimator_max_depth': 4, 'model_n_estimators': 220, 'model_learning_rate': 0.025} |
| AdaBoost (With FE) | 0.673 ± 0.015 | 0.682 ± 0.012 | 0.683 | -0.001 | {'model_estimator_max_depth': 4, 'model_n_estimators': 140, 'model_learning_rate': 0.152} |

| Model | Default: CV mean ± std | Tuned: nested CV mean ± std | Tuning score (best trial) | Optimism | Chosen hyperparameters |
|---|---|---|---|---|---|
| Logistic Regression (Without FE) | 0.672 ± 0.013 | 0.673 ± 0.016 | 0.678 | +0.002 | {'model_C': 0.0715} |
| Logistic Regression (With FE) | 0.675 ± 0.016 | 0.673 ± 0.014 | 0.676 | +0.001 | {'model_C': 0.1586} |

Conclusions:

- Across all three model families, feature engineering yielded no significant performance gains that exceeded the fold-to-fold standard deviation (between 0.012–0.017), meaning no model family truly "gained" in terms of final nested CV accuracy. 

- Instead, tree-based models maintained stable scores while optimizing internal structures and efficiency (more balanced depths/leafs for RF, fewer estimators for AdaBoost), while the linear model only benefited from a more relaxed regularization parameter (C).

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////