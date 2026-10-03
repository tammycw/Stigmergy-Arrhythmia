# Stigmergy-Arrhythmia

A termite-inspired ECG beat classification project that compares two evaluation settings:

- interpatient classification using patient-disjoint train/test records
- intrapatient classification using a standard within-dataset split

The model is built around a two-stage stigmergic pipeline: first separating Normal beats from Abnormal beats, and then distinguishing Ventricular (V) versus Fusion (F) beats in the abnormal subgroup.

## Overview

This repository contains two complementary workflows for arrhythmia classification using the MIT-BIH Arrhythmia Database:

- `Stigmergy-Arrhythmia-twostage-inter.py` implements the interpatient setup.
- `Stigmergy-Arrhythmia-twostage-intra.py` implements the intrapatient setup.

Both approaches use a swarm-inspired pheromone representation, PCA-based embedding, and a KNN classifier trained on swarm-derived features. The key difference is how the data are split for evaluation.

## Interpatient vs. intrapatient evaluation

### 1. Interpatient workflow

File: `Stigmergy-Arrhythmia-twostage-inter.py`

This setup follows the AAMI patient-disjoint protocol. The training and test sets are split at the record level, so beats from the same patient never appear in both partitions.

Why it matters:
- It better reflects real-world generalization to new patients.
- It reduces leakage from patient-specific morphology.
- It is the more rigorous benchmark for ECG beat classification.

The script uses AAMI DS1 and DS2 records to enforce this separation and is designed for a patient-level generalization test.

### 2. Intrapatient workflow

File: `Stigmergy-Arrhythmia-twostage-intra.py`

This setup uses a train/test split without enforcing patient-disjoint records. It is useful for exploratory experiments and internal comparisons, but it is usually less strict than the interpatient setup because information from the same patient can appear in both train and test folds.

This variant is still valuable for understanding model behavior and for comparing raw-vs-derived feature pipelines.

## Two-stage classification

Both scripts use a staged class hierarchy:

### Stage 1: Normal vs. Abnormal

- Train a binary classifier to separate Normal (N) beats from abnormal beats.
- The abnormal group includes V, F, and S beats.
- The learned pheromone field and swarm descriptors are used to predict whether each beat is normal or abnormal.

### Stage 2: Abnormal subtype classification

- Keep only the abnormal beats that are feasible for subtype modeling.
- Train a second classifier to distinguish Ventricular (V) beats from Fusion (F) beats.
- Supraventricular (S) beats are intentionally excluded from this stage because their sample count is too low for a stable subtype model.

This keeps the broad diagnosis separate from the more focused subtype decision and makes the overall pipeline more interpretable.

## Why the swarm formulation is useful

The project models ECG beats as autonomous termite-like agents that deposit class-specific pheromone patterns in a 2D lattice. Over time, local interactions create a collective spatial field that reflects class structure. The final classifier is then trained on summary statistics derived from this pheromone neighborhood rather than using only raw feature values directly.

The design goals are:
- class separation via local swarm dynamics
- explicit spatial interpretation of learned class structure
- robustness to class imbalance
- a modular two-stage architecture for binary and subtype decision making

## Repository structure

```text
Stigmergy-Arrhythmia/
├── README.md
├── Stigmergy_download_wfdb_to_excel.py
├── Stigmergy_compute_derived_features.py
├── Stigmergy-Arrhythmia-twostage-inter.py
├── Stigmergy-Arrhythmia-twostage-intra.py
├── input/
│   └── ... dataset files
├── module_analysis/
│   └── ... analysis utilities
├── output-stigmergy-twostage-interpatient-derived/
├── output-stigmergy-twostage-interpatient-raw/
├── output-stigmergy-twostage-intrapatient-derived/
├── output-stigmergy-twostage-intrapatient-raw/
└── ...
```

## Data preparation

The project expects ECG data stored in Excel workbooks generated from MIT-BIH data.

Typical workflow:

1. Download PhysioNet ECG records.
2. Build feature tables and metadata.
3. Filter low-quality RR rows when applicable.
4. Run either the interpatient or intrapatient pipeline.
5. Save metrics, counts, plots, and predictions to the output folder.

## Run the workflows

### Interpatient run

```bash
python Stigmergy-Arrhythmia-twostage-inter.py
```

### Intrapatient run

```bash
python Stigmergy-Arrhythmia-twostage-intra.py
```

## Typical outputs

The scripts generate:

- class-count summaries
- AAMI filtered partitions for the interpatient workflow
- confusion matrices
- ROC curves and metric plots
- heatmaps and pheromone evolution output
- Excel files with predictions
- JSON summaries of performance and class distributions

## Practical interpretation

- Use the interpatient workflow when you want a realistic and conservative patient-generalization benchmark.
- Use the intrapatient workflow for exploratory analysis or quick model comparisons.
- Report the interpatient result as the primary benchmark when comparing to clinical deployment settings.

## Installation

Requirements are approximately:

```bash
pip install numpy pandas scikit-learn scipy wfdb plotnine openpyxl
```

## Scientific context

This work is inspired by stigmergic colony behavior, where local interactions produce global structure without a central controller. In the ECG setting, the swarm acts as a learned spatial representation of class-specific beat organization, and the final decisions are made using the emergent pheromone landscape rather than a purely direct threshold-based rule.

## Notes

- The project is designed around MIT-BIH arrhythmia data and AAMI beat labels.
- Stage 2 intentionally excludes S beats to avoid unstable subtype training from underrepresented data.
- Output folders are separated by dataset choice and evaluation protocol to keep results organized and reproducible.

## Citation

If you use this work in a project or paper, cite the repository and clearly state which protocol was used: interpatient or intrapatient.

## Author

Tammy Wang

Email: tammycc.wang@gmail.com

Oklahoma School of Science and Mathematics (OSSM)

- MIT-BIH Arrhythmia Database from PhysioNet
- scikit-learn and pandas communities
- plotnine developers for ggplot2-style visualization in Python

---

**Questions or Contributions?** Feel free to open an issue or submit a pull request!
