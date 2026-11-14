# Bias mitigation with the Disparate Impact Remover (Adult, German Credit, COMPAS)

This repository contains the code and results of an assignment on bias mitigation. The chosen method is the **Disparate Impact Remover** (Feldman et al., 2015, *Certifying and Removing Disparate Impact*), a **pre-processing** algorithm. It is applied to three datasets (Adult, German Credit, COMPAS).

* **Task 2:** DP_Diff, DIR (disparate impact ratio) and consistency on the original labels, before mitigation.
* **Task 3:** stratified train/test split, repair of the training data, logistic regression, and evaluation on the held-out test set (DP_Diff, DIR, AEO_Diff, PP_Diff, consistency) before vs. after mitigation, plus a sensitivity analysis over the repair level λ.

## Repository structure

```
.
├── assignment-4.ipynb      # the whole analysis (already executed, outputs are saved in the notebook)
├── requirements.txt        # Python dependencies
├── README.md
└── data/
    ├── adult.csv                         # Adult census income
    ├── german.data                       # German credit (original categorical version)
    ├── german.doc                        # attribute description of the German data
    └── cox-violent-parsed_filt.csv       # COMPAS (ProPublica, Cox/violent version, filtered columns)
```

## Data

| Dataset | File used | Source |
|---|---|---|
| Adult | `data/adult.csv` | https://www.kaggle.com/datasets/uciml/adult-census-income |
| German Credit | `data/german.data` | https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data |
| COMPAS | `data/cox-violent-parsed_filt.csv` | https://www.kaggle.com/datasets/danofer/compass |

The data files are copies of the public datasets from the sources above and keep their original licences.

Preparation (done in the notebook):
* **Favorable label (Y = 1):** Adult income > 50K; German good credit risk; COMPAS no recidivism (`is_recid` = 0).
* **COMPAS:** standard ProPublica filters (`days_b_screening_arrest` between -30 and 30, `is_recid` != -1, ordinary charge degree), one row per person (`id`).
* **Protected attributes (privileged / unprivileged):** Adult sex (Male / Female) and race (White / non-White); German age (>= 25 / < 25) and sex (male / female, from *personal status and sex*); COMPAS race (Caucasian / non-Caucasian) and sex (Female / Male).
* **Features:** categorical columns one-hot encoded, numeric columns min-max scaled; protected attributes are not used as classifier inputs.

## How to reproduce

Tested with Python 3.12 and 3.13.

```bash
# 1. get the code
git clone <URL-OF-THIS-REPOSITORY>
cd <REPOSITORY-FOLDER>

# 2. create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. install the dependencies
pip install -r requirements.txt

# 4a. run the notebook interactively
jupyter notebook assignment-4.ipynb        # then: Run -> Run All Cells

# 4b. or run it from the command line (overwrites the saved outputs with identical ones)
jupyter nbconvert --to notebook --execute --inplace assignment-4.ipynb
```

The run takes about one minute. All paths are relative (`data/`), so the notebook has to be started from the repository root. The warnings printed when `aif360` is imported (missing `tensorflow`, `fairlearn`, `inFairness`) are harmless: those optional packages are not needed.

## Settings

| Setting | Value |
|---|---|
| Method | Disparate Impact Remover (AIF360 implementation) |
| Repair level λ | 1.0 in the main comparison; 0, 0.25, 0.5, 0.75, 1.0 in the sensitivity analysis |
| Split | 70 % train / 30 % test, stratified by label x protected group, `random_state = 42` |
| Classifier | logistic regression, `max_iter = 2000`, `random_state = 42` |
| Consistency | k = 5 nearest neighbours, computed on the original test features |
| Feature option | `NUMERIC_ONLY = False` (all features; `True` keeps only numeric features) |

The results are deterministic (fixed seed), so a re-run gives the same numbers.

## Results

**Original labels (before any classifier)**

| Dataset | Protected attribute (privileged / unprivileged) | N | DP_Diff | DIR | Consistency |
|---|---|---|---|---|---|
| Adult | sex (Male / Female) | 32561 | -0.196 | 0.358 | 0.793 |
| Adult | race (White / non-White) | 32561 | -0.103 | 0.596 | 0.793 |
| German | age (≥ 25 / < 25) | 1000 | -0.129 | 0.821 | 0.672 |
| German | sex (male / female) | 1000 | -0.075 | 0.897 | 0.672 |
| COMPAS | race (Caucasian / non-Caucasian) | 8891 | -0.088 | 0.847 | 0.648 |
| COMPAS | sex (Female / Male) | 8891 | -0.173 | 0.736 | 0.648 |

**Test set, before → after the Disparate Impact Remover (λ = 1.0)**

| Dataset | Protected | DP_Diff | DIR | AEO_Diff | PP_Diff | Consistency |
|---|---|---|---|---|---|---|
| Adult | sex | -0.170 → -0.163 | 0.322 → 0.331 | -0.093 → -0.086 | -0.082 → -0.090 | 0.942 → 0.944 |
| Adult | race | -0.092 → -0.074 | 0.558 → 0.634 | -0.047 → -0.017 | -0.027 → -0.058 | 0.944 → 0.945 |
| German | age | -0.228 → -0.039 | 0.705 → 0.957 | -0.148 → 0.011 | -0.198 → -0.170 | 0.779 → 0.880 |
| German | sex | 0.000 → -0.056 | 1.000 → 0.934 | 0.028 → -0.046 | -0.060 → -0.032 | 0.780 → 0.797 |
| COMPAS | race | -0.181 → -0.134 | 0.756 → 0.809 | -0.158 → -0.110 | -0.020 → -0.030 | 0.984 → 0.962 |
| COMPAS | sex | -0.184 → -0.066 | 0.748 → 0.894 | -0.145 → -0.024 | -0.091 → -0.113 | 0.985 → 0.952 |

(DP_Diff, AEO_Diff, PP_Diff: 0 is fair; DIR: 1 is fair; consistency: higher is better. Differences are unprivileged − privileged.) The notebook also contains the sensitivity analysis over λ (DIR, AEO_Diff and accuracy).

## Notes

* The AIF360 implementation has no separate transform step, so the test features are repaired using their own distributions.
* Results come from a single split and seed; the German test set is small (300 rows), so its values are noisy.

## Reference

M. Feldman, S. A. Friedler, J. Moeller, C. Scheidegger, S. Venkatasubramanian. *Certifying and Removing Disparate Impact.* KDD 2015.
