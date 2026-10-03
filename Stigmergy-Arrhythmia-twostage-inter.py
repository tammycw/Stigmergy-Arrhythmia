# Tammy Wang | OSSM | 06/15/2026 | tammycc.wang@gmail.com
"""
Updated: 08/23/2026; 10/02/2026
================================================================================
STIGMERGY-ARRHYTHMIA: Stigmergic Termite-Inspired ECG Beat Classification
================================================================================

Scientific context
------------------
This script implements an inter-patient ECG beat classification pipeline using the
AAMI (Association for the Advancement of Medical Instrumentation) DS1/DS2
partition, in accordance with the standard AAMI EC57 patient-disjoint evaluation
protocol described in:

ANSI/AAMI EC57:2012/(R)2020, Testing and reporting performance results of cardiac
rhythm and ST segment measurement algorithms.
https://doi.org/10.2345/9781570204784.ch1

The key requirement is that training and test sets are separated at the patient
record level: beats from the same patient record never appear in both the training
and test partitions. This prevents record-level leakage and is the standard
benchmarking setup for ECG arrhythmia classification.

The model is inspired by stigmergic behavior in natural termite colonies: local
agents deposit and react to pheromone fields in a shared environment, and global
structure emerges from local interactions rather than a single centralized
controller. The same idea is applied here to ECG beat classification:

1. Each beat is represented as an autonomous termite agent.
2. Agents move through a 2D lattice and interact with class-specific pheromone
   fields.
3. Pheromone diffusion and evaporation smooth the field over time.
4. Local colony neighborhoods encode class-relevant spatial context.
5. Classification emerges from the resulting swarm response rather than from a
   direct rule-based threshold.

Method overview
---------------
The workflow uses a two-stage hierarchical classifier under an AAMI inter-patient
split. Each stage has its own pheromone field, swarm-derived descriptors, KNN
classifier, and patient-disjoint train/test partition. The pipeline first separates
Normal beats from Abnormal beats, then performs a focused V-versus-F subtype task
for the abnormal class.

1. Data preparation
   - load raw waveforms or derived features and remove rows flagged as RR-quality
     problems
   - use the AAMI record partition (DS1 for training, DS2 for testing)
   - enforce record-level separation so no patient appears in both folds
   - fit StandardScaler and PCA on DS1 only, then transform both partitions to
     avoid test-set leakage

2. Stigmergic representation
   - map each beat's PCA coordinates to a 2D lattice
   - move class-labeled termite agents toward their class centroid and toward
     matching local pheromone while repelling opposing-class pheromone
   - deposit class-weighted pheromone and apply Gaussian diffusion and
     evaporation to produce smooth class fields
   - summarize each beat's local neighborhood using the mean and standard
     deviation of the pheromone response

3. Stage 1: Normal vs. Abnormal
   - train a binary pheromone field and KNN classifier on DS1 records
   - evaluate on the held-out DS2 records
   - focus on N vs. abnormal discrimination before subtype analysis

4. Stage 2: Abnormal subtype classification
   - retain only ventricular (V) and fusion (F) beats
   - intentionally exclude supraventricular (S) beats because their sample count
     is too small to support a reliable subtype model
   - train a separate V-versus-F pheromone field and KNN classifier under the
     same patient-disjoint split
   - assess subtype performance independently from Stage 1

5. Evaluation and output
   - report accuracy, precision, recall, F1, Cohen's Kappa, AUROC, and
     confusion matrices for both Train and Test
   - save class counts, filtered AAMI records, metrics, plots, and predictions
     for reproducibility

Leakage and imbalance safeguards
--------------------------------
- DS1 and DS2 are fixed by record ID, so beats from one patient record cannot
  appear in both Train and Test.
- The AAMI record lists are screened after RR-quality filtering and checked for
  overlap before model fitting.
- StandardScaler and PCA are fit on DS1 only.
- Pheromone deposition upweights the minority class in each binary stage.
- When enabled, minority training samples may be oversampled before KNN fitting;
  evaluation always uses the original train/test data.
- Stage 2 excludes S instead of training an unstable subtype model from a tiny
  class count.

Core functions
--------------
- resolve_dataset_path(...): locate the selected Excel workbook
- load_excel_dataset(...): load features and metadata while filtering rows with
  RR-quality issues when applicable
- TermiteAgent.move(...): local stigmergic colony movement
- train_pheromone_classifier(...): self-organizing pheromone training
- extract_swarm_features(...): convert the colony field into class descriptors
- predict_with_pheromone(...): backward-compatible pheromone score output
- evaluate_split(...): compute accuracy, precision, recall, F1, Kappa, and AUROC

Outputs
-------
- PCA scatter plot of predictions versus truth
- ROC comparison plot
- performance comparison plot
- pheromone evolution animation
- pheromone heatmaps
- confusion-matrix plots for training and testing
- Excel workbook with train/test predictions
- Stage 1 binary metrics and predictions
- Stage 2 V-versus-F subtype metrics and predictions, with S excluded

================================================================================
"""

import os
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.neighbors import KNeighborsClassifier
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation, PillowWriter
from plotnine import ggplot, aes, geom_point, geom_line, geom_abline, geom_text, geom_col, labs, theme_minimal, theme, ggsave, scale_color_manual, scale_fill_manual, scale_x_continuous, scale_y_continuous, coord_flip, position_dodge
from plotnine.themes.elements import element_blank, element_line, element_rect, element_text


plt.rcParams.update({
    'font.family': 'Arial',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'sans-serif'],
    'axes.titleweight': 'bold',
})

current_dir = Path(__file__).resolve().parent

# ------------------ Input Data configuration ------------------
# Choose which Excel workbook to use for model training:
# - "raw" : use the raw beat-waveform workbook
# - "derived" : use the derived-feature workbook, filtered to rows marked OK
DATASET_CHOICE = "derived"  # "raw" or "derived"
# For inter-patient ECG splits, aggressive reweighting can make the training
# data look artificially easy while hurting DS2 generalization. Keep this
# conservative by default and tune it on validation instead of the final test set.
HANDLE_CLASS_IMBALANCE = False
# Prevent aggressive duplication of rare samples when balancing KNN training.
# A value of 1.0 disables oversampling; values above 1.0 gradually increase the
# minority class. Keep it moderate to avoid overfitting patient-specific beats.
MAX_MINORITY_OVERSAMPLING_RATIO = 1.5
RAW_DATASET_CANDIDATES = [
    current_dir / "input" / "mitbih_raw_data_rrflag.xlsx",
]
DERIVED_DATASET_CANDIDATES = [
    current_dir / "input" / "derived_waveform_features.xlsx",
]

# Directory for all generated outputs (plots, Excel, logs)
output_dir = current_dir / f'output-stigmergy-twostage-interpatient-{DATASET_CHOICE}'
output_dir.mkdir(parents=True, exist_ok=True)
log_path = output_dir / "stigmergy_run.log"
log_messages = []


def log_message(message):
    text = str(message)
    print(text)
    log_messages.append(text)


timer_state = {}


def start_timer(name):
    timer_state[name] = time.perf_counter()


def stop_timer(name):
    if name not in timer_state:
        return
    elapsed = time.perf_counter() - timer_state[name]
    log_message(f"TIMER [{name}]: {elapsed:.3f} seconds")
    timer_state[name] = elapsed


def log_timer_summary():
    log_message("\n=== Timing Summary ===")
    for name, value in timer_state.items():
        if isinstance(value, float):
            log_message(f"{name}: {value:.3f} seconds")


log_message(f"Current directory: {current_dir}")
start_timer("overall")



# ------------------ Dataset split configuration ------------------
# The AAMI inter-patient split uses DS1 records for training and DS2 records
# for testing. Beats from one record cannot cross Train and Test.
SPLIT_PRESET = "aami"  # "70_30", "50_50", "30_70", or "aami"
AAMI_DS1_RECORDS = [
    101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
    201, 203, 205, 207, 208, 209, 215, 220, 223, 230,
]
AAMI_DS2_RECORDS = [
    100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212,
    213, 214, 219, 221, 222, 228, 231, 234,
]
SPLIT_PRESETS = {
    "70_30": (0.70, 0.30),
    "50_50": (0.50, 0.50),
    "30_70": (0.30, 0.70),
    "aami": (None, None),
}

if SPLIT_PRESET not in SPLIT_PRESETS:
    raise ValueError(f"Unknown SPLIT_PRESET '{SPLIT_PRESET}'. Choose one of: {list(SPLIT_PRESETS.keys())}")

if SPLIT_PRESET == "aami":
    train_fraction, test_fraction = None, None
    log_message(
        f"Dataset split preset: {SPLIT_PRESET} | "
        f"DS1 train={len(AAMI_DS1_RECORDS)} records, DS2 test={len(AAMI_DS2_RECORDS)} records"
    )
else:
    train_fraction, test_fraction = SPLIT_PRESETS[SPLIT_PRESET]
    if not np.isclose(train_fraction + test_fraction, 1.0):
        raise ValueError("Training and testing fractions must sum to 1.0.")
    log_message(
        f"Dataset split preset: {SPLIT_PRESET} | "
        f"train={train_fraction:.2f}, test={test_fraction:.2f}"
    )



# ====================== 1. Load Excel Data ======================
def resolve_dataset_path(dataset_choice):
    if dataset_choice == "raw":
        candidates = RAW_DATASET_CANDIDATES
    elif dataset_choice == "derived":
        candidates = DERIVED_DATASET_CANDIDATES
    else:
        raise ValueError("DATASET_CHOICE must be 'raw' or 'derived'")

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(f"No workbook found for dataset '{dataset_choice}'. Looked for: {', '.join(str(p) for p in candidates)}")


def load_excel_dataset(dataset_choice):
    excel_path = resolve_dataset_path(dataset_choice)
    log_message(f"Loading {dataset_choice} dataset from {excel_path}")

    features = pd.read_excel(excel_path, sheet_name="features")
    metadata = pd.read_excel(excel_path, sheet_name="metadata")

    if dataset_choice == "derived":
        # Derived datasets may contain rows that should be excluded because the
        # RR-based metadata flags indicate a problem. Keep only rows marked OK.
        flag_column = None
        for candidate in ("rr_problem_flag", "rr_problem_flg"):
            if candidate in metadata.columns:
                flag_column = candidate
                break

        if flag_column is None:
            log_message("Derived dataset selected but no RR problem flag column was found in metadata; keeping all rows")
        else:
            keep_mask = metadata[flag_column].astype(str).str.strip().str.lower() == "ok"
            kept_rows = int(keep_mask.sum())
            dropped_rows = int((~keep_mask).sum())
            features = features.loc[keep_mask].reset_index(drop=True)
            metadata = metadata.loc[keep_mask].reset_index(drop=True)
            log_message(
                f"Derived dataset filter: kept {kept_rows} rows and dropped {dropped_rows} rows where {flag_column} != 'ok'"
            )

    if dataset_choice == "raw":
        labels = pd.read_excel(excel_path, sheet_name="labels")
        flag_column = None
        for candidate in ("rr_problem_flg", "rr_problem_flag"):
            if candidate in metadata.columns:
                flag_column = candidate
                break

        if flag_column is None:
            log_message("Raw dataset selected but no RR problem flag column was found in metadata; keeping all rows")
        else:
            keep_mask = metadata[flag_column].astype(str).str.strip().str.lower() == "ok"
            kept_rows = int(keep_mask.sum())
            dropped_rows = int((~keep_mask).sum())
            features = features.loc[keep_mask].reset_index(drop=True)
            metadata = metadata.loc[keep_mask].reset_index(drop=True)
            labels = labels.loc[keep_mask].reset_index(drop=True)
            log_message(
                f"Raw dataset filter: kept {kept_rows} rows and dropped {dropped_rows} rows where {flag_column} != 'ok'"
            )

    features = features.copy()
    features = features.apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = features.to_numpy(dtype=np.float32)
    record_ids = metadata["record_id"].to_numpy(dtype=int)
    beat_classes = metadata["beat_class"].astype(str).str.strip().str.upper().to_numpy(dtype=object)
    supported_classes = {"N", "V", "S", "F"}
    unsupported_classes = sorted(set(beat_classes) - supported_classes)
    if unsupported_classes:
        raise ValueError(f"Unsupported beat classes found: {unsupported_classes}")
    y = (beat_classes != "N").astype(np.float32).reshape(-1, 1)
    feature_names = [str(col) for col in features.columns]
    return X, y, record_ids, beat_classes, feature_names


def split_interpatient_indices(record_ids, labels, test_fraction, random_state=42):
    """Split complete records using an integer training-record count."""
    record_ids = np.asarray(record_ids)
    labels = np.asarray(labels, dtype=int)
    unique_records = np.unique(record_ids)
    n_records = len(unique_records)
    n_train_records = int(n_records * (1.0 - test_fraction))
    n_test_records = n_records - n_train_records
    if n_train_records < 1 or n_test_records < 1:
        raise ValueError(
            f"Split requires at least one train and test record; got "
            f"train={n_train_records}, test={n_test_records}."
        )
    record_labels = np.asarray([
        np.bincount(labels[record_ids == record_id]).argmax()
        for record_id in unique_records
    ])

    stratify = record_labels if len(np.unique(record_labels)) > 1 else None
    try:
        train_records, test_records = train_test_split(
            unique_records,
            train_size=n_train_records,
            test_size=n_test_records,
            stratify=stratify,
            random_state=random_state,
        )
    except ValueError:
        train_records, test_records = train_test_split(
            unique_records,
            train_size=n_train_records,
            test_size=n_test_records,
            stratify=None,
            random_state=random_state,
        )

    train_idx = np.flatnonzero(np.isin(record_ids, train_records))
    test_idx = np.flatnonzero(np.isin(record_ids, test_records))
    if np.intersect1d(record_ids[train_idx], record_ids[test_idx]).size:
        raise RuntimeError("Inter-patient split contains overlapping records.")
    return train_idx, test_idx


def normalize_by_record(X, record_ids):
    """Apply per-record z-scoring to reduce patient-to-patient feature drift."""
    X_norm = X.astype(np.float32, copy=True)
    for record_id in np.unique(record_ids):
        mask = record_ids == record_id
        if np.sum(mask) == 0:
            continue
        record_mean = X_norm[mask].mean(axis=0)
        record_std = X_norm[mask].std(axis=0)
        record_std = np.where(record_std < 1e-8, 1.0, record_std)
        X_norm[mask] = (X_norm[mask] - record_mean) / record_std
    return X_norm


def filter_low_drift_features(X_train, X_test, y_train, y_test, keep_fraction=0.75):
    """Keep the most stable features across train/test partitions to reduce domain shift."""
    if X_train.shape[1] == 0:
        return X_train, X_test, np.array([], dtype=bool)

    class_labels = sorted(np.unique(np.concatenate([y_train, y_test])).astype(int).tolist())
    drift_scores = []
    for feature_idx in range(X_train.shape[1]):
        class_shifts = []
        for class_label in class_labels:
            train_mask = y_train == class_label
            test_mask = y_test == class_label
            if np.any(train_mask) and np.any(test_mask):
                train_mean = float(np.mean(X_train[train_mask, feature_idx]))
                test_mean = float(np.mean(X_test[test_mask, feature_idx]))
                class_shifts.append(abs(train_mean - test_mean))
        if class_shifts:
            drift_scores.append(float(np.mean(class_shifts)))
        else:
            drift_scores.append(0.0)

    drift_scores = np.asarray(drift_scores, dtype=np.float32)
    if drift_scores.size == 0:
        return X_train, X_test, np.ones(X_train.shape[1], dtype=bool)

    threshold = np.quantile(drift_scores, keep_fraction)
    keep_mask = drift_scores <= threshold
    if np.count_nonzero(keep_mask) == 0:
        keep_mask = np.ones_like(drift_scores, dtype=bool)

    log_message(
        f"Feature drift filter retained {np.count_nonzero(keep_mask)}/{len(keep_mask)} features "
        f"(keep_fraction={keep_fraction}, threshold={threshold:.4f})"
    )
    return X_train[:, keep_mask], X_test[:, keep_mask], keep_mask


start_timer("data_loading")
X, y, record_ids, beat_classes, feature_names = load_excel_dataset(DATASET_CHOICE)
log_message(f"Loaded {DATASET_CHOICE} dataset with shape {X.shape}")
beat_class_counts = {
    class_name: int(np.sum(beat_classes == class_name))
    for class_name in sorted(set(beat_classes))
}
log_message(f"Beat classes available: {list(beat_class_counts)}")
log_message(f"Beat class counts: {beat_class_counts}")
with open(output_dir / 'beat_class_counts.json', 'w', encoding='utf-8') as handle:
    json.dump(beat_class_counts, handle, indent=2)
stop_timer("data_loading")

if SPLIT_PRESET == "aami":
    available_record_ids = set(np.unique(record_ids).astype(int).tolist())
    AAMI_DS1_RECORDS = sorted(record_id for record_id in AAMI_DS1_RECORDS if record_id in available_record_ids)
    AAMI_DS2_RECORDS = sorted(record_id for record_id in AAMI_DS2_RECORDS if record_id in available_record_ids)
    if set(AAMI_DS1_RECORDS) & set(AAMI_DS2_RECORDS):
        raise ValueError("AAMI DS1 and DS2 record partitions must not overlap.")
    if not AAMI_DS1_RECORDS or not AAMI_DS2_RECORDS:
        raise ValueError("Filtered AAMI DS1 and DS2 partitions must both contain records.")
    log_message(f"Filtered AAMI DS1 training records ({len(AAMI_DS1_RECORDS)}): {AAMI_DS1_RECORDS}")
    log_message(f"Filtered AAMI DS2 test records ({len(AAMI_DS2_RECORDS)}): {AAMI_DS2_RECORDS}")
    filtered_aami_partitions = {
        "DS1_train": AAMI_DS1_RECORDS,
        "DS2_test": AAMI_DS2_RECORDS,
    }
    with open(output_dir / 'aami_filtered_partitions.json', 'w', encoding='utf-8') as handle:
        json.dump(filtered_aami_partitions, handle, indent=2)
else:
    filtered_aami_partitions = None

y_true = y.reshape(-1)
stage1_sample_sizes = {
    "Normal": int(np.sum(y_true == 0)),
    "Abnormal": int(np.sum(y_true == 1)),
}
stage1_sample_sizes["Total"] = stage1_sample_sizes["Normal"] + stage1_sample_sizes["Abnormal"]
log_message(f"Stage 1 total sample sizes: {stage1_sample_sizes}")

# Stage 1 uses patient-disjoint AAMI partitions whenever the AAMI preset is
# active; all beats from DS1 records train the model and all beats from DS2
# records are held out for testing.
if SPLIT_PRESET == "aami":
    ds1_mask = np.isin(record_ids, AAMI_DS1_RECORDS)
    ds2_mask = np.isin(record_ids, AAMI_DS2_RECORDS)
    train_idx = np.where(ds1_mask)[0]
    test_idx = np.where(ds2_mask)[0]
else:
    # Fractional presets also remain patient-disjoint; only the record ratio
    # changes between 70/30, 50/50, and 30/70.
    train_idx, test_idx = split_interpatient_indices(
        record_ids,
        y_true,
        test_fraction=test_fraction,
        random_state=42,
    )

stage1_train_records = sorted(np.unique(record_ids[train_idx]).astype(int).tolist())
stage1_test_records = sorted(np.unique(record_ids[test_idx]).astype(int).tolist())
log_message(
    f"Stage 1 training records ({len(stage1_train_records)}): {stage1_train_records}"
)
log_message(
    f"Stage 1 testing records ({len(stage1_test_records)}): {stage1_test_records}"
)

X_train = X[train_idx]
X_test = X[test_idx]
y_train = y_true[train_idx]
y_test = y_true[test_idx]
X_train, X_test, stage1_keep_mask = filter_low_drift_features(X_train, X_test, y_train, y_test, keep_fraction=0.75)
X = X[:, stage1_keep_mask]
X_train = X[train_idx]
X_test = X[test_idx]
X = normalize_by_record(X, record_ids)
X_train = X[train_idx]
X_test = X[test_idx]

start_timer("preprocessing")
scaler = StandardScaler()
if SPLIT_PRESET == "aami":
    X_scaled = scaler.fit(X[ds1_mask]).transform(X)
else:
    X_scaled = scaler.fit_transform(X)

# A true termite colony needs a 2D embedding on which agents can self-organize.
# PCA is therefore used as the lattice coordinate system only; the final label
# emerges from the pheromone field and colony dominance rather than from a
# PCA-based decision threshold.
pca = PCA(n_components=2, random_state=42)
if SPLIT_PRESET == "aami":
    pca.fit(X_scaled[ds1_mask])
    X_2d = pca.transform(X_scaled)
else:
    X_2d = pca.fit_transform(X_scaled)
log_message(f"Data shape: {X.shape} | Standardized feature space: {X_scaled.shape} | Reduced visualization: {X_2d.shape}")
stop_timer("preprocessing")

X_train = X_scaled[train_idx]
X_test = X_scaled[test_idx]
y_train = y_true[train_idx]
y_test = y_true[test_idx]
X_train_2d = X_2d[train_idx]
X_test_2d = X_2d[test_idx]
stage1_split_sample_sizes = {
    "Train": {
        "Normal": int(np.sum(y_train == 0)),
        "Abnormal": int(np.sum(y_train == 1)),
        "Total": int(len(y_train)),
    },
    "Test": {
        "Normal": int(np.sum(y_test == 0)),
        "Abnormal": int(np.sum(y_test == 1)),
        "Total": int(len(y_test)),
    },
}
log_message(f"Stage 1 Train/Test category sample sizes: {stage1_split_sample_sizes}")

# ====================== 2. Termite-Inspired Swarm Model ======================

class TermiteAgent:
    """
    A termite scout that carries one ECG beat into the shared colony field.

    The scout now uses two forces during movement:
    1. attraction to the class-specific colony center; and
    2. local attraction/repulsion based on the current pheromone landscape.

    This produces a true self-organizing colony process where class-specific
    clusters emerge through local interactions, not through global PCA thresholding.
    """

    def __init__(self, position, sample_point, class_label, class_center):
        self.pos = position
        self.carrying = sample_point
        self.class_label = int(class_label)
        self.class_center = class_center.astype(float)
        self.pheromone_memory = 0.0

    def move(self, grid_size, pheromone_grids):
        """
        Move by local stigmergic bias and class-center attraction.
        """
        x, y = self.pos.astype(int)

        # Bias toward the class colony center in the 2D lattice.
        center_dx = self.class_center[0] - x
        center_dy = self.class_center[1] - y
        dx = int(np.sign(center_dx)) if center_dx != 0 else 0
        dy = int(np.sign(center_dy)) if center_dy != 0 else 0

        neighbor_candidates = []
        for m in (-1, 0, 1):
            for n in (-1, 0, 1):
                if m == 0 and n == 0:
                    continue
                nx = int(np.clip(x + m, 0, grid_size - 1))
                ny = int(np.clip(y + n, 0, grid_size - 1))
                same = pheromone_grids[self.class_label, nx, ny]
                other = pheromone_grids[1 - self.class_label, nx, ny]
                score = same - 0.35 * other
                neighbor_candidates.append(((m, n), score))

        if neighbor_candidates:
            best_dir, _ = max(neighbor_candidates, key=lambda item: item[1])
            if best_dir[0] != 0 or best_dir[1] != 0:
                dx = int(best_dir[0])
                dy = int(best_dir[1])

        # Add a small exploratory step so the colony can still reconfigure.
        if np.random.rand() < 0.15:
            dx += np.random.choice([-1, 0, 1])
            dy += np.random.choice([-1, 0, 1])

        self.pos = np.clip(self.pos + [dx, dy], 0, grid_size - 1)



def build_grid_bounds(X):
    """
    Compute the 2D lattice bounds used to map PCA coordinates into the
    stigmergy grid.
    """
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-8)
    return mins, spans



def scale_to_grid(point, grid_size, bounds):
    """
    Map a 2D PCA embedding into the pheromone lattice.
    """
    mins, spans = bounds
    indices = np.floor((point - mins) / spans * (grid_size - 1)).astype(int)
    return np.clip(indices, 0, grid_size - 1).astype(float)


def balance_training_data(X_train, y_train, random_state=42, max_minority_ratio=3.0):
    """Oversample minority classes up to a configurable minority ratio."""
    y_train = np.asarray(y_train, dtype=int)
    class_labels, class_counts = np.unique(y_train, return_counts=True)
    if len(class_labels) < 2:
        raise ValueError("Training data must contain both classes.")

    rng = np.random.default_rng(random_state)
    target_count = int(min(class_counts.max(), class_counts.min() * max_minority_ratio))
    balanced_indices = []
    for class_label, class_count in zip(class_labels, class_counts):
        class_indices = np.flatnonzero(y_train == class_label)
        balanced_indices.extend(
            rng.choice(class_indices, size=target_count, replace=class_count < target_count)
        )

    balanced_indices = np.asarray(balanced_indices, dtype=int)
    rng.shuffle(balanced_indices)
    return X_train[balanced_indices], y_train[balanced_indices]


def prepare_knn_training_data(X_train, y_train, handle_class_imbalance, random_state=42):
    """Scale augmented features, then optionally oversample DS1 training data."""
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    if handle_class_imbalance:
        X_train_scaled, y_train = balance_training_data(
            X_train_scaled,
            y_train,
            random_state=random_state,
            max_minority_ratio=MAX_MINORITY_OVERSAMPLING_RATIO,
        )
    return feature_scaler, X_train_scaled, y_train


def train_pheromone_classifier(X_train, y_train, grid_size=60, n_iterations=260, record_snapshots=False, snapshot_interval=20, handle_class_imbalance=True):
    pheromone_grids = np.zeros((2, grid_size, grid_size), dtype=np.float32)
    y_train = np.asarray(y_train, dtype=int)
    bounds = build_grid_bounds(X_train)

    class_counts = np.bincount(y_train, minlength=2)
    if handle_class_imbalance:
        max_count = max(class_counts.max(), 1)
        raw_class_weights = {
            class_label: max_count / max(class_counts[class_label], 1)
            for class_label in range(2)
        }
        # Keep minority-class emphasis modest for inter-patient ECG data; a large
        # raw ratio (e.g. 9x) can make the training split appear perfect while badly
        # overfitting the source-domain morphology.
        class_weights = {
            class_label: min(2.5, raw_class_weights[class_label])
            for class_label in range(2)
        }
    else:
        class_weights = {0: 1.0, 1: 1.0}
    log_message(f"Pheromone class weights: {class_weights}")

    class_centroids = {
        0: scale_to_grid(X_train[y_train == 0].mean(axis=0), grid_size, bounds),
        1: scale_to_grid(X_train[y_train == 1].mean(axis=0), grid_size, bounds),
    }

    agents = [
        TermiteAgent(scale_to_grid(point, grid_size, bounds), point, label, class_centroids[label])
        for point, label in zip(X_train, y_train)
    ]

    snapshots = []
    if record_snapshots:
        snapshots.append((0, pheromone_grids.copy()))

    log_message("Running termite-inspired stigmergy training...")
    for it in range(n_iterations):
        for agent in agents:
            agent.move(grid_size, pheromone_grids)
            x_idx, y_idx = agent.pos.astype(int)
            deposit_strength = 1.2 * class_weights[agent.class_label]
            pheromone_grids[agent.class_label, x_idx, y_idx] += deposit_strength

        pheromone_grids[0] = gaussian_filter(pheromone_grids[0], sigma=1.2)
        pheromone_grids[1] = gaussian_filter(pheromone_grids[1], sigma=1.2)
        pheromone_grids *= 0.982

        if record_snapshots and (it + 1) % snapshot_interval == 0:
            snapshots.append((it + 1, pheromone_grids.copy()))

        if it % 50 == 0:
            log_message(f"Iter {it:3d} | Normal pheromone: {pheromone_grids[0].mean():.4f} | Abnormal pheromone: {pheromone_grids[1].mean():.4f}")

    for cls in range(2):
        max_val = float(np.max(pheromone_grids[cls]))
        if max_val > 0:
            pheromone_grids[cls] = pheromone_grids[cls] / max_val

    if record_snapshots:
        return pheromone_grids, bounds, snapshots
    return pheromone_grids, bounds


def extract_swarm_features(X_eval, pheromone_grids, grid_size=60, bounds=None):
    """
    Convert the learned pheromone field into explicit per-sample features.

    Each sample receives a local colony descriptor based on the pheromone
    neighborhood around its lattice position. These descriptors provide the
    differentiating signal that the final KNN classifier can use.
    """
    features = []
    for point in X_eval:
        pos = scale_to_grid(point, grid_size, bounds)
        x_idx, y_idx = int(pos[0]), int(pos[1])

        local_values = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx = int(np.clip(x_idx + dx, 0, grid_size - 1))
                ny = int(np.clip(y_idx + dy, 0, grid_size - 1))
                normal_score = pheromone_grids[0, nx, ny]
                abnormal_score = pheromone_grids[1, nx, ny]
                local_values.append([normal_score, abnormal_score, abnormal_score - normal_score])

        local_array = np.array(local_values, dtype=np.float32)
        features.append([
            local_array[:, 0].mean(),
            local_array[:, 1].mean(),
            local_array[:, 2].mean(),
            local_array[:, 0].std(),
            local_array[:, 1].std(),
            local_array[:, 2].std(),
        ])
    return np.array(features, dtype=np.float32)


def predict_with_pheromone(X_eval, pheromone_grids, grid_size=60, bounds=None):
    """
    Backward-compatible pheromone score extractor used for visualization and AUROC.
    """
    predicted = []
    score_diffs = []
    for point in X_eval:
        pos = scale_to_grid(point, grid_size, bounds)
        x_idx, y_idx = int(pos[0]), int(pos[1])

        local_patch = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx = int(np.clip(x_idx + dx, 0, grid_size - 1))
                ny = int(np.clip(y_idx + dy, 0, grid_size - 1))
                local_patch.append((pheromone_grids[0, nx, ny], pheromone_grids[1, nx, ny]))

        normal_score = np.mean([item[0] for item in local_patch])
        abnormal_score = np.mean([item[1] for item in local_patch])
        score_diff = abnormal_score - normal_score
        score_diffs.append(score_diff)
        predicted.append(int(np.argmax([normal_score, abnormal_score])))
    return np.array(predicted, dtype=int), np.array(score_diffs, dtype=float)


def save_pheromone_animation(snapshots, output_path):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    titles = ["Normal pheromone field", "Abnormal pheromone field"]

    images = []
    for idx, (title, ax) in enumerate(zip(titles, axs)):
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        images.append(ax.imshow(snapshots[0][1][idx], cmap='viridis', vmin=0, vmax=1, origin='lower'))

    fig.suptitle(f"Pheromone evolution: iteration {snapshots[0][0]}")

    def update(frame_index):
        iteration, grids = snapshots[frame_index]
        images[0].set_data(grids[0])
        images[1].set_data(grids[1])
        fig.suptitle(f"Pheromone evolution: iteration {iteration}")
        return images

    anim = FuncAnimation(fig, update, frames=len(snapshots), interval=800, blit=False)
    writer = PillowWriter(fps=1)
    anim.save(str(output_path), writer=writer)
    plt.close(fig)


def save_pheromone_heatmaps(pheromone_grids, output_path):
    diff_grid = pheromone_grids[1] - pheromone_grids[0]
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["Normal class", "Abnormal class", "Abnormal - Normal"]
    grids = [pheromone_grids[0], pheromone_grids[1], diff_grid]
    cmaps = ['Blues', 'Reds', 'coolwarm']

    for ax, grid, title, cmap in zip(axs, grids, titles, cmaps):
        im = ax.imshow(grid, cmap=cmap, origin='lower')
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle('Final Pheromone Fields and Class Difference')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(output_path), dpi=200)
    plt.close(fig)


def save_predictor_workbook(train_matrix, test_matrix, feature_names, output_path, sheet_prefix='stage1'):
    """Write the exact train/test predictor matrices used by the classifier."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    train_df = pd.DataFrame(train_matrix, columns=feature_names)
    test_df = pd.DataFrame(test_matrix, columns=feature_names)
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        train_df.to_excel(writer, sheet_name=f'{sheet_prefix}_train', index=False)
        test_df.to_excel(writer, sheet_name=f'{sheet_prefix}_test', index=False)
        pd.DataFrame({'feature_name': feature_names}).to_excel(
            writer,
            sheet_name=f'{sheet_prefix}_feature_names',
            index=False,
        )
    log_message(f"Saved predictor matrices to {output_path}")


def save_confusion_matrix_figure(confusion_matrices, labels, title, output_path, split_names=None):
    soft_blue = LinearSegmentedColormap.from_list(
        'soft_blue', ['#f8fbff', '#d4e6f8', '#6ca8db']
    )
    split_names = split_names or ['Training', 'Testing']
    n_matrices = len(confusion_matrices)
    n_cols = min(3, n_matrices)
    n_rows = int(np.ceil(n_matrices / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.8 * n_rows), constrained_layout=True)
    if n_matrices == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).ravel()
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.05)
    fig.subplots_adjust(top=0.86)

    for ax, cm_values, split_name in zip(axes, confusion_matrices, split_names):
        cm = np.asarray(cm_values)
        total = cm.sum()
        normalized = cm.astype(float) / total if total else cm.astype(float)

        ax.imshow(normalized, cmap=soft_blue, vmin=0, vmax=1, alpha=0.8, zorder=1)
        ax.set_title(split_name, fontsize=12, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels, fontsize=10, color='#222222')
        ax.set_yticklabels(labels, fontsize=10, color='#222222')
        ax.set_xlabel('Predicted label', fontsize=10, color='#222222')
        ax.set_ylabel('True label', fontsize=10, color='#222222')
        ax.tick_params(length=0)
        ax.set_aspect('equal')
        ax.set_facecolor('#f8f7f3')
        ax.set_frame_on(True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#444444')
            spine.set_linewidth(1.2)

        cell_colors = [
            ['#f3f1ee', '#d7e6f5'],
            ['#f5dfdc', '#e8e8e4'],
        ]
        for i in range(2):
            for j in range(2):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           facecolor=cell_colors[i][j],
                                           edgecolor='none',
                                           zorder=0))

        ax.set_xticks(np.arange(cm.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(cm.shape[0] + 1) - 0.5, minor=True)
        ax.grid(which='minor', color='white', linewidth=1)
        ax.tick_params(which='minor', length=0)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                value = cm[i, j]
                percent = 100.0 * value / total if total else 0.0
                ax.text(j, i, f'{value}\n[{percent:.1f}%]', ha='center', va='center',
                        color='black', fontsize=10, fontweight='bold', linespacing=1.2)

    for ax in axes[n_matrices:]:
        ax.axis('off')

    fig.savefig(str(output_path), dpi=300, bbox_inches='tight')
    plt.close(fig)


def get_trained_artifacts():
    """Return trained objects and data slices for downstream analysis.

    This helper allows external analysis scripts to import the trained
    classifier, preprocessing pipeline, pheromone field, and test data
    without re-running training logic manually.
    """
    swarm_feature_names = [
        'Normal_mean','Abnormal_mean','Difference_mean',
        'Normal_std','Abnormal_std','Difference_std'
    ]
    artifacts = {
        'scaler': scaler,
        'pca': pca,
        'pheromone_grids': pheromone_grids,
        'pheromone_bounds': pheromone_bounds,
        'pheromone_snapshots': pheromone_snapshots,
        'knn_classifier': knn_classifier,
        'X_train': X_train,
        'X_train_aug': X_train_aug,
        'y_train': y_train,
        'X_test_2d': X_test_2d,
        'X_test': X_test,
        'X_test_aug': X_test_aug,
        'y_test': y_test,
        'test_scores': test_scores,
        'feature_names': feature_names + swarm_feature_names,
    }
    return artifacts


# ====================== 3. Stage 1: Normal vs Abnormal ======================
log_message("Start Stage 1 model training: Normal vs Abnormal...")
log_message(f"Class imbalance handling enabled: {HANDLE_CLASS_IMBALANCE}")
start_timer("model_training")
grid_size = 60
pheromone_grids, pheromone_bounds, pheromone_snapshots = train_pheromone_classifier(
    X_train_2d,
    y_train,
    grid_size=grid_size,
    record_snapshots=True,
    snapshot_interval=20,
    handle_class_imbalance=HANDLE_CLASS_IMBALANCE,
)

# Build one feature vector per beat from the emergent pheromone field.
train_swarm_features = extract_swarm_features(
    X_train_2d,
    pheromone_grids,
    grid_size=grid_size,
    bounds=pheromone_bounds,
)
test_swarm_features = extract_swarm_features(
    X_test_2d,
    pheromone_grids,
    grid_size=grid_size,
    bounds=pheromone_bounds,
)

# Let the pristine colony field act as a learned feature extractor, and then use
# a stable k-nearest neighbors (KNN) classifier to resolve labels.

# Do NOT incorporate waveform-derived features into the model at this stage.
# Keep swarm-derived augmentation only.
# Save the exact predictor matrices used by Stage 1 for downstream inspection.
stage1_feature_names = [
    feature_names[idx]
    for idx in range(len(feature_names))
    if stage1_keep_mask[idx]
] + [
    'Normal_mean', 'Abnormal_mean', 'Difference_mean',
    'Normal_std', 'Abnormal_std', 'Difference_std',
]
X_train_aug = np.hstack([X_train, train_swarm_features])
X_test_aug = np.hstack([X_test, test_swarm_features])
stage1_predictor_path = output_dir / 'stage1_predictor_matrix.xlsx'
save_predictor_workbook(
    X_train_aug,
    X_test_aug,
    stage1_feature_names,
    stage1_predictor_path,
    sheet_prefix='stage1',
)
stage1_knn_scaler, X_knn_train, y_knn_train = prepare_knn_training_data(
    X_train_aug,
    y_train,
    HANDLE_CLASS_IMBALANCE,
)
X_knn_test = stage1_knn_scaler.transform(X_test_aug)
log_message(f"Stage 1 KNN training samples after preparation: {len(y_knn_train)}")

knn_classifier = KNeighborsClassifier(n_neighbors=9, weights='distance')
knn_classifier.fit(X_knn_train, y_knn_train)
X_knn_train_original = stage1_knn_scaler.transform(X_train_aug)
train_pred = knn_classifier.predict(X_knn_train_original)
test_pred = knn_classifier.predict(X_knn_test)
train_scores = knn_classifier.predict_proba(X_knn_train_original)[:, 1]
test_scores = knn_classifier.predict_proba(X_knn_test)[:, 1]
stop_timer("model_training")

# ====================== 4. Evaluation ======================
def evaluate_split(y_true, y_pred, scores, label):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "auroc": roc_auc_score(y_true, scores),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
    log_message(f"=== {label} ===")
    log_message(f"Accuracy: {metrics['accuracy']:.3f}")
    log_message(f"Precision (abnormal class): {metrics['precision']:.3f}")
    log_message(f"Recall (abnormal class): {metrics['recall']:.3f}")
    log_message(f"F1-score (abnormal class): {metrics['f1']:.3f}")
    log_message(f"Cohen's Kappa: {metrics['kappa']:.3f}")
    log_message(f"AUROC: {metrics['auroc']:.3f}")
    log_message("Confusion Matrix:\n" + str(metrics['confusion_matrix']))
    return metrics


def select_probability_threshold(y_true, scores, thresholds=None):
    """Pick a probability cutoff that maximizes F1 for the abnormal class."""
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)
    thresholds = np.asarray(thresholds, dtype=np.float32)
    best_threshold = 0.5
    best_metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    best_recall = -1.0
    for threshold in thresholds:
        y_pred = (scores >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if (f1 > best_metrics['f1']) or (np.isclose(f1, best_metrics['f1']) and recall > best_recall):
            best_threshold = float(threshold)
            best_metrics = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
            }
            best_recall = float(recall)
    return best_threshold, best_metrics


def search_stage2_configuration(X_train, y_true, scores, threshold_grid=None, confidence_grid=None, min_per_class=0):
    """Search for the highest-F1 Stage 2 configuration without enforcing class preservation during optimization."""
    if threshold_grid is None:
        threshold_grid = np.linspace(0.15, 0.85, 15)
    if confidence_grid is None:
        confidence_grid = np.linspace(0.0, 0.90, 19)
    threshold_grid = np.asarray(threshold_grid, dtype=np.float32)
    confidence_grid = np.asarray(confidence_grid, dtype=np.float32)

    best = {
        'confidence': 0.0,
        'decision_threshold': 0.5,
        'precision': 0.0,
        'recall': 0.0,
        'f1': -1.0,
        'n_train': 0,
    }

    for conf_threshold in confidence_grid:
        keep_mask = scores >= conf_threshold
        if not np.any(keep_mask):
            continue

        kept_counts = [int(np.sum(keep_mask[y_true == cls])) for cls in range(2)]
        if min(kept_counts) < min_per_class:
            continue

        filtered_X = X_train[keep_mask]
        filtered_y = y_true[keep_mask]
        if len(filtered_y) < 10 or len(np.unique(filtered_y)) < 2:
            continue

        candidate_knn = KNeighborsClassifier(
            n_neighbors=min(9, len(filtered_y)),
            weights='distance',
        )
        candidate_knn.fit(filtered_X, filtered_y)
        filtered_scores = candidate_knn.predict_proba(filtered_X)[:, 1]
        decision_threshold, metrics = select_probability_threshold(
            filtered_y,
            filtered_scores,
            thresholds=threshold_grid,
        )
        pred = (filtered_scores >= decision_threshold).astype(int)
        f1 = f1_score(filtered_y, pred, zero_division=0)
        if f1 > best['f1'] or (np.isclose(f1, best['f1']) and metrics['recall'] > best['recall']):
            best = {
                'confidence': float(conf_threshold),
                'decision_threshold': float(decision_threshold),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1': float(f1),
                'n_train': int(len(filtered_y)),
            }

    if best['f1'] < 0:
        return np.ones_like(y_true, dtype=bool), 0.5, 0.0
    return scores >= best['confidence'], best['decision_threshold'], best['confidence']


training_metrics = evaluate_split(y_train, train_pred, train_scores, "Training")
testing_metrics = evaluate_split(y_test, test_pred, test_scores, "Testing")

log_message("\n=== Training vs Testing Summary ===")
for metric_name in ["accuracy", "precision", "recall", "f1", "kappa", "auroc"]:
    log_message(
        f"{metric_name}: training={training_metrics[metric_name]:.3f}, testing={testing_metrics[metric_name]:.3f}"
    )

split_order = ['Train', 'Test']

metric_comparison_df = pd.DataFrame({
    'metric': ['Accuracy', 'Precision', 'Recall', 'F1', 'Kappa', 'AUROC'],
    'Train': [training_metrics['accuracy'], training_metrics['precision'], training_metrics['recall'], training_metrics['f1'], training_metrics['kappa'], training_metrics['auroc']],
    'Test': [testing_metrics['accuracy'], testing_metrics['precision'], testing_metrics['recall'], testing_metrics['f1'], testing_metrics['kappa'], testing_metrics['auroc']],
})
metric_comparison_long = pd.melt(
    metric_comparison_df,
    id_vars=['metric'],
    value_vars=split_order,
    var_name='split',
    value_name='value'
)
metric_comparison_long['split'] = pd.Categorical(
    metric_comparison_long['split'],
    categories=split_order,
    ordered=True,
)
metrics_plot = (
    ggplot(metric_comparison_long, aes(x='metric', y='value', fill='split'))
    + geom_col(position=position_dodge(), width=0.8)
    + labs(
        title='Performance Summary Across Train and Test Splits',
        x='Metric',
        y='Score',
        fill='Split'
    )
    + scale_fill_manual(values=['#0072B2', '#009E73'])
    + scale_y_continuous(breaks=list(np.arange(0, 1.01, 0.1)), limits=(0, 1.0), expand=(0, 0))
    + theme_minimal(base_size=13)
    + theme(
        plot_title=element_text(hjust=0.5, size=16, weight='bold'),
        legend_position='bottom',
        legend_title=element_text(size=11, weight='bold'),
        legend_text=element_text(size=10),
        panel_grid_major_y=element_blank(),
        panel_grid_minor=element_blank(),
        axis_line=element_line(color='black'),
        panel_border=element_rect(color='black', fill=None, size=1),
        axis_ticks=element_line(color='black', size=0.6)
    )
)

# Visualization with ggplot2-style grammar of graphics
viz_df = pd.DataFrame({
    'x': X_test_2d[:, 0],
    'y': X_test_2d[:, 1],
    'actual': np.where(y_test == 0, 'Normal', 'Abnormal'),
    'predicted': np.where(test_pred == 0, 'Normal', 'Abnormal')
})

plot = (
    ggplot(viz_df, aes(x='x', y='y', color='predicted', shape='actual'))
    + geom_point(alpha=0.72, size=2.7)
    + labs(
        title='Termite-Inspired Stigmergy Classifier on ECG Beat Embeddings',
        x='PCA Component 1',
        y='PCA Component 2',
        color='Predicted Label',
        shape='True Label'
    )
    + scale_color_manual(values=['#1f77b4', '#d62728'])
    + theme_minimal(base_size=13)
    + theme(
        plot_title=element_text(hjust=0.5, size=16, weight='bold'),
        legend_position='bottom',
        legend_title=element_text(size=11, weight='bold'),
        legend_text=element_text(size=10),
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        axis_line=element_line(color='black'),
        panel_border=element_rect(color='black', fill=None, size=1),
        axis_ticks=element_line(color='black', size=0.6)
    )
)

fpr_train, tpr_train, _ = roc_curve(y_train, train_scores)
fpr_test, tpr_test, _ = roc_curve(y_test, test_scores)
roc_df = pd.DataFrame({
    'fpr': np.concatenate([fpr_train, fpr_test]),
    'tpr': np.concatenate([tpr_train, tpr_test]),
    'split': np.concatenate([
        np.repeat('Train', len(fpr_train)),
        np.repeat('Test', len(fpr_test))
    ]),
    'auroc': np.concatenate([
        np.repeat(training_metrics['auroc'], len(fpr_train)),
        np.repeat(testing_metrics['auroc'], len(fpr_test))
    ])
})
roc_df['split'] = pd.Categorical(
    roc_df['split'],
    categories=split_order,
    ordered=True,
)
annotation_df = pd.DataFrame({
    'split': ['Train', 'Test'],
    'x': [0.72, 0.72],
    'y': [0.12, 0.06],
    'label': [
        f'Train AUROC = {training_metrics["auroc"]:.2f}',
        f'Test AUROC = {testing_metrics["auroc"]:.2f}'
    ]
})
roc_plot = (
    ggplot(roc_df, aes(x='fpr', y='tpr', color='split'))
    + geom_line(size=1.15)
    + geom_abline(intercept=0, slope=1, linetype='dashed', color='gray', size=0.8)
    + geom_text(
        annotation_df,
        aes(x='x', y='y', label='label'),
        ha='left',
        size=10,
        color='black',
        show_legend=False
    )
    + labs(
        title='ROC Curves for Train and Test Splits',
        x='False Positive Rate',
        y='True Positive Rate',
        color='Split'
    )
    + scale_color_manual(values=['#1f77b4', '#ff7f0e', '#d62728'])
    + scale_x_continuous(limits=(0, 1), expand=(0, 0))
    + scale_y_continuous(limits=(0, 1), expand=(0, 0))
    + theme_minimal(base_size=13)
    + theme(
        plot_title=element_text(hjust=0.5, size=16, weight='bold'),
        legend_position='bottom',
        legend_title=element_text(size=11, weight='bold'),
        legend_text=element_text(size=10),
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        axis_line=element_line(color='black'),
        panel_border=element_rect(color='black', fill=None, size=1),
        axis_ticks=element_line(color='black', size=0.6)
    )
)

pr_curve_frames = []
for split_name, y_true, scores in [
    ('Train', y_train, train_scores),
    ('Test', y_test, test_scores),
]:
    precision, recall, _ = precision_recall_curve(y_true, scores)
    pr_curve_frames.append(
        pd.DataFrame({
            'precision': precision,
            'recall': recall,
            'split': split_name,
        })
    )
pr_curve_df = pd.concat(pr_curve_frames, ignore_index=True)
pr_curve_df['split'] = pd.Categorical(
    pr_curve_df['split'],
    categories=split_order,
    ordered=True,
)
pr_curve_plot = (
    ggplot(pr_curve_df, aes(x='recall', y='precision', color='split'))
    + geom_line(size=1.1)
    + labs(
        title='Precision–Recall Curves for the Abnormal Class',
        x='Recall',
        y='Precision',
        color='Split'
    )
    + scale_color_manual(values=['#1f77b4', '#ff7f0e', '#d62728'])
    + scale_x_continuous(limits=(0, 1), expand=(0, 0))
    + scale_y_continuous(limits=(0, 1), expand=(0, 0))
    + theme_minimal(base_size=13)
    + theme(
        plot_title=element_text(hjust=0.5, size=16, weight='bold'),
        legend_position='bottom',
        legend_title=element_text(size=11, weight='bold'),
        legend_text=element_text(size=10),
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        axis_line=element_line(color='black'),
        panel_border=element_rect(color='black', fill=None, size=1),
        axis_ticks=element_line(color='black', size=0.6)
    )
)

swarm_feature_df = pd.DataFrame(
    test_swarm_features,
    columns=[
        'Normal_mean',
        'Abnormal_mean',
        'Difference_mean',
        'Normal_std',
        'Abnormal_std',
        'Difference_std',
    ]
)
swarm_feature_df['actual'] = np.where(y_test == 0, 'Normal', 'Abnormal')

swarm_feature_distribution_fig, swarm_axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
swarm_feature_distribution_fig.suptitle('Distribution of Swarm-Derived Features in the Held-Out Test Set', fontsize=14, fontweight='bold')
feature_columns = swarm_feature_df.columns[:-1]

for ax, feature_name in zip(swarm_axes.ravel(), feature_columns):
    normal_values = swarm_feature_df.loc[swarm_feature_df['actual'] == 'Normal', feature_name]
    abnormal_values = swarm_feature_df.loc[swarm_feature_df['actual'] == 'Abnormal', feature_name]

    box = ax.boxplot(
        [normal_values, abnormal_values],
        patch_artist=True,
        widths=0.5,
        showfliers=False,
        medianprops={'color': '#000000', 'linewidth': 1.2},
        boxprops={'linewidth': 0.9, 'edgecolor': '#333333'},
        whiskerprops={'color': '#333333', 'linewidth': 0.9},
        capprops={'color': '#333333', 'linewidth': 0.9},
    )

    for patch, color in zip(box['boxes'], ['#4c78a8', '#d62728']):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Normal', 'Abnormal'], fontsize=10)
    ax.set_title(feature_name.replace('_', ' '), fontsize=10, fontweight='bold')
    ax.set_ylabel('Feature value', fontsize=10)
    ax.tick_params(axis='both', labelsize=9)
    ax.grid(False)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

swarm_feature_distribution_fig.align_ylabels()

pdf_path = output_dir / 'stigmergy_clustering.pdf'
roc_pdf_path = output_dir / 'stigmergy_auroc.pdf'
metrics_pdf_path = output_dir / 'stigmergy_metrics_comparison.pdf'
pr_pdf_path = output_dir / 'stigmergy_precision_recall.pdf'
swarm_distribution_pdf_path = output_dir / 'stigmergy_swarm_feature_distribution.pdf'
process_gif_path = output_dir / 'stigmergy_pheromone_evolution.gif'
pheromone_heatmap_path = output_dir / 'stigmergy_pheromone_heatmap.pdf'
confusion_matrix_path = output_dir / 'stigmergy_confusion_matrices.pdf'
prediction_path = output_dir / 'stigmergy_predictions.xlsx'
plot.save(filename=str(pdf_path), width=10, height=8, dpi=300)
roc_plot.save(filename=str(roc_pdf_path), width=8, height=6, dpi=300)
metrics_plot.save(filename=str(metrics_pdf_path), width=8, height=6, dpi=300)
pr_curve_plot.save(filename=str(pr_pdf_path), width=8, height=6, dpi=300)
swarm_feature_distribution_fig.savefig(str(swarm_distribution_pdf_path), dpi=300, bbox_inches='tight')
save_confusion_matrix_figure(
    [training_metrics['confusion_matrix'], testing_metrics['confusion_matrix']],
    ['Normal', 'Abnormal'],
    'Confusion Matrices for Train and Test Splits',
    confusion_matrix_path,
    split_names=['Train', 'Test'],
)
save_pheromone_animation(pheromone_snapshots, process_gif_path)
save_pheromone_heatmaps(pheromone_grids, pheromone_heatmap_path)
stop_timer("visualization")

training_prediction_df = pd.DataFrame({
    'prediction': train_pred,
    'observation': y_train
})
testing_prediction_df = pd.DataFrame({
    'prediction': test_pred,
    'observation': y_test
})

with pd.ExcelWriter(prediction_path) as writer:
    training_prediction_df.to_excel(writer, sheet_name='training', index=False)
    testing_prediction_df.to_excel(writer, sheet_name='testing', index=False)

log_message(f'Saved visualization to {pdf_path}')
log_message(f'Saved AUROC plot to {roc_pdf_path}')
log_message(f'Saved metrics comparison plot to {metrics_pdf_path}')
log_message(f'Saved precision-recall plot to {pr_pdf_path}')
log_message(f'Saved swarm-feature distribution figure to {swarm_distribution_pdf_path}')
log_message(f'Saved confusion matrix figure to {confusion_matrix_path}')
log_message(f'Saved predictions to {prediction_path}')

# ====================== 5. Stage 2: Abnormal subtype V vs F ======================
# Train a separate subtype model on V/F beats only; S beats are intentionally
# excluded because their sample count is too low for reliable subtype training.
log_message("\nStart Stage 2 model training: Abnormal subtype V vs F...")
subtype_mask = np.isin(beat_classes, ["V", "F"])
ignored_subtype_count = int(np.sum(beat_classes == "S"))
stage2_total_sample_sizes = {
    "V": int(np.sum(beat_classes == "V")),
    "F": int(np.sum(beat_classes == "F")),
    "S": ignored_subtype_count,
}
stage2_classification_sample_sizes = {
    "V": stage2_total_sample_sizes["V"],
    "F": stage2_total_sample_sizes["F"],
}
stage2_classification_total = sum(stage2_classification_sample_sizes.values())
log_message(f"Stage 2 total sample sizes: {stage2_total_sample_sizes}")
log_message(
    "Stage 2 total sample sizes for classification: "
    f"{stage2_classification_sample_sizes} (Total V + F: {stage2_classification_total})"
)
log_message(f"Ignoring {ignored_subtype_count} S beats in Stage 2 subtype training.")

subtype_classes = beat_classes[subtype_mask]
subtype_label_map = {"V": 0, "F": 1}
subtype_label_names = {0: "V", 1: "F"}
subtype_y = np.array([subtype_label_map[label] for label in subtype_classes], dtype=int)
if set(subtype_classes) != {"V", "F"}:
    raise ValueError("Stage 2 requires both V and F beats; S is ignored.")

subtype_X_scaled = X_scaled[subtype_mask]
subtype_X_2d = X_2d[subtype_mask]
subtype_record_ids = record_ids[subtype_mask]
# Reuse the Stage 1 patient partition so the hierarchy evaluates both stages
# on the same held-out records. Stage 2 trains a fresh V/F pheromone field.
subtype_train_idx = np.where(np.isin(subtype_record_ids, stage1_train_records))[0]
subtype_test_idx = np.where(np.isin(subtype_record_ids, stage1_test_records))[0]
stage2_train_records = sorted(np.unique(subtype_record_ids[subtype_train_idx]).astype(int).tolist())
stage2_test_records = sorted(np.unique(subtype_record_ids[subtype_test_idx]).astype(int).tolist())
log_message(
    f"Stage 2 training records ({len(stage2_train_records)}): {stage2_train_records}"
)
log_message(
    f"Stage 2 testing records ({len(stage2_test_records)}): {stage2_test_records}"
)
subtype_X_train_raw = subtype_X_scaled[subtype_train_idx]
subtype_X_test_raw = subtype_X_scaled[subtype_test_idx]
subtype_y_train = subtype_y[subtype_train_idx]
subtype_y_test = subtype_y[subtype_test_idx]
subtype_X_train_raw, subtype_X_test_raw, subtype_feature_mask = filter_low_drift_features(
    subtype_X_train_raw,
    subtype_X_test_raw,
    subtype_y_train,
    subtype_y_test,
    keep_fraction=0.65,
)
stage2_base_feature_names = [
    feature_names[idx]
    for idx in range(len(feature_names))
    if stage1_keep_mask[idx]
]
subtype_X_scaled = subtype_X_scaled[:, subtype_feature_mask]
subtype_X_2d = subtype_X_2d  # PCA coordinates remain tied to the filtered stage-1 feature space
subtype_X_train_2d = subtype_X_2d[subtype_train_idx]
subtype_X_test_2d = subtype_X_2d[subtype_test_idx]
stage2_split_sample_sizes = {
    "Train": {
        "V": int(np.sum(subtype_y_train == subtype_label_map["V"])),
        "F": int(np.sum(subtype_y_train == subtype_label_map["F"])),
        "S": 0,
        "Total": int(len(subtype_y_train)),
    },
    "Test": {
        "V": int(np.sum(subtype_y_test == subtype_label_map["V"])),
        "F": int(np.sum(subtype_y_test == subtype_label_map["F"])),
        "S": 0,
        "Total": int(len(subtype_y_test)),
    },
}
log_message(f"Stage 2 Train/Test category sample sizes: {stage2_split_sample_sizes}")

subtype_grid_size = 60
subtype_pheromone_grids, subtype_pheromone_bounds = train_pheromone_classifier(
    subtype_X_train_2d,
    subtype_y_train,
    grid_size=subtype_grid_size,
)
subtype_train_swarm_features = extract_swarm_features(
    subtype_X_train_2d,
    subtype_pheromone_grids,
    grid_size=subtype_grid_size,
    bounds=subtype_pheromone_bounds,
)
subtype_test_swarm_features = extract_swarm_features(
    subtype_X_test_2d,
    subtype_pheromone_grids,
    grid_size=subtype_grid_size,
    bounds=subtype_pheromone_bounds,
)
subtype_X_train_aug = np.hstack([
    subtype_X_scaled[subtype_train_idx],
    subtype_train_swarm_features,
])
subtype_X_test_aug = np.hstack([
    subtype_X_scaled[subtype_test_idx],
    subtype_test_swarm_features,
])
subtype_feature_names = [
    stage2_base_feature_names[idx]
    for idx in range(len(stage2_base_feature_names))
    if subtype_feature_mask[idx]
] + [
    'Normal_mean', 'Abnormal_mean', 'Difference_mean',
    'Normal_std', 'Abnormal_std', 'Difference_std',
]
stage2_predictor_path = output_dir / 'stage2_predictor_matrix.xlsx'
save_predictor_workbook(
    subtype_X_train_aug,
    subtype_X_test_aug,
    subtype_feature_names,
    stage2_predictor_path,
    sheet_prefix='stage2',
)
subtype_knn_scaler, subtype_X_knn_train, subtype_y_knn_train = prepare_knn_training_data(
    subtype_X_train_aug,
    subtype_y_train,
    HANDLE_CLASS_IMBALANCE,
    random_state=43,
)
subtype_X_knn_test = subtype_knn_scaler.transform(subtype_X_test_aug)
log_message(f"Stage 2 KNN training samples after preparation: {len(subtype_y_knn_train)}")

subtype_knn_classifier = KNeighborsClassifier(
    n_neighbors=min(9, len(subtype_y_train)),
    weights='distance',
)
subtype_knn_classifier.fit(subtype_X_knn_train, subtype_y_knn_train)
subtype_X_knn_train_original = subtype_knn_scaler.transform(subtype_X_train_aug)
subtype_train_scores = subtype_knn_classifier.predict_proba(subtype_X_knn_train_original)[:, 1]
subtype_test_scores = subtype_knn_classifier.predict_proba(subtype_X_knn_test)[:, 1]

subtype_thresholds = np.linspace(0.15, 0.85, 15)
subtype_best_threshold, subtype_best_metrics = select_probability_threshold(
    subtype_y_train,
    subtype_train_scores,
    thresholds=subtype_thresholds,
)
subtype_X_train_aug_full = subtype_X_train_aug.copy()
subtype_y_train_full = subtype_y_train.copy()
confident_train_mask, subtype_best_threshold, conf_filter_threshold = search_stage2_configuration(
    subtype_X_knn_train,
    subtype_y_knn_train,
    subtype_train_scores,
    threshold_grid=subtype_thresholds,
    confidence_grid=np.linspace(0.0, 0.90, 19),
    min_per_class=0,
)
subtype_X_train_aug = subtype_X_train_aug[confident_train_mask]
subtype_y_train = subtype_y_train[confident_train_mask]
if np.unique(subtype_y_train).size < 2:
    log_message(
        "Stage 2 search produced a single-class training subset; restoring the full training set for valid probability estimation."
    )
    subtype_X_train_aug = subtype_X_train_aug_full
    subtype_y_train = subtype_y_train_full
subtype_X_knn_train = subtype_knn_scaler.transform(subtype_X_train_aug)
subtype_knn_classifier = KNeighborsClassifier(
    n_neighbors=min(9, len(subtype_y_train)),
    weights='distance',
)
subtype_knn_classifier.fit(subtype_X_knn_train, subtype_y_train)
subtype_X_knn_train_original = subtype_knn_scaler.transform(subtype_X_train_aug)
subtype_train_pred = subtype_knn_classifier.predict(subtype_X_knn_train_original)
subtype_test_pred = subtype_knn_classifier.predict(subtype_X_knn_test)
subtype_train_scores = subtype_knn_classifier.predict_proba(subtype_X_knn_train_original)[:, 1]
subtype_test_scores = subtype_knn_classifier.predict_proba(subtype_X_knn_test)[:, 1]
subtype_best_threshold, subtype_best_metrics = select_probability_threshold(
    subtype_y_train,
    subtype_train_scores,
    thresholds=subtype_thresholds,
)
log_message(
    "Stage 2 tuned threshold: "
    f"thr={subtype_best_threshold:.3f} | precision={subtype_best_metrics['precision']:.3f} | "
    f"recall={subtype_best_metrics['recall']:.3f} | f1={subtype_best_metrics['f1']:.3f} | "
    f"confidence_filter={conf_filter_threshold:.3f} | kept_train={len(subtype_y_train)}"
)
subtype_train_pred = (subtype_train_scores >= subtype_best_threshold).astype(int)
subtype_test_pred = (subtype_test_scores >= subtype_best_threshold).astype(int)

subtype_training_metrics = evaluate_split(
    subtype_y_train,
    subtype_train_pred,
    subtype_train_scores,
    "Stage 2 Training (V vs F)",
)
subtype_testing_metrics = evaluate_split(
    subtype_y_test,
    subtype_test_pred,
    subtype_test_scores,
    "Stage 2 Testing (V vs F)",
)
subtype_metrics = {
    "training": subtype_training_metrics,
    "test": subtype_testing_metrics,
    "ignored_class": "S",
}

subtype_metric_comparison_df = pd.DataFrame({
    'metric': ['Accuracy', 'Precision', 'Recall', 'F1', 'Kappa', 'AUROC'],
    'Train': [
        subtype_training_metrics['accuracy'],
        subtype_training_metrics['precision'],
        subtype_training_metrics['recall'],
        subtype_training_metrics['f1'],
        subtype_training_metrics['kappa'],
        subtype_training_metrics['auroc'],
    ],
    'Test': [
        subtype_testing_metrics['accuracy'],
        subtype_testing_metrics['precision'],
        subtype_testing_metrics['recall'],
        subtype_testing_metrics['f1'],
        subtype_testing_metrics['kappa'],
        subtype_testing_metrics['auroc'],
    ],
})
subtype_metric_comparison_long = pd.melt(
    subtype_metric_comparison_df,
    id_vars=['metric'],
    value_vars=['Train', 'Test'],
    var_name='split',
    value_name='value',
)
subtype_metric_comparison_long['split'] = pd.Categorical(
    subtype_metric_comparison_long['split'],
    categories=['Train', 'Test'],
    ordered=True,
)
subtype_metrics_plot = (
    ggplot(subtype_metric_comparison_long, aes(x='metric', y='value', fill='split'))
    + geom_col(position=position_dodge(), width=0.8)
    + labs(
        title='Abnormal Subtype Performance: V vs F',
        x='Metric',
        y='Score',
        fill='Split',
    )
    + scale_fill_manual(values=['#0072B2', '#D55E00'])
    + scale_y_continuous(
        breaks=list(np.arange(0, 1.01, 0.1)),
        limits=(0, 1.0),
        expand=(0, 0),
    )
    + theme_minimal(base_size=13)
    + theme(
        plot_title=element_text(hjust=0.5, size=16, weight='bold'),
        legend_position='bottom',
        legend_title=element_text(size=11, weight='bold'),
        legend_text=element_text(size=10),
        panel_grid_major_y=element_blank(),
        panel_grid_minor=element_blank(),
        axis_line=element_line(color='black'),
        panel_border=element_rect(color='black', fill=None, size=1),
        axis_ticks=element_line(color='black', size=0.6),
    )
)
subtype_metrics_pdf_path = output_dir / 'abnormal_subtype_metrics_comparison.pdf'
subtype_metrics_plot.save(
    filename=str(subtype_metrics_pdf_path),
    width=8,
    height=6,
    dpi=300,
)
log_message(f"Saved abnormal subtype metrics comparison plot to {subtype_metrics_pdf_path}")

save_confusion_matrix_figure(
    [
        subtype_training_metrics['confusion_matrix'],
        subtype_testing_metrics['confusion_matrix'],
    ],
    ['V', 'F'],
    'Abnormal Subtype Confusion Matrices (V vs F; S ignored)',
    output_dir / 'abnormal_subtype_confusion_matrices.pdf',
    split_names=['Train', 'Test'],
)
subtype_predictions = [
    pd.DataFrame({
        'record_id': record_ids[subtype_mask][subtype_train_idx],
        'true_label': [subtype_classes[index] for index in subtype_train_idx],
        'predicted_label': [subtype_label_names[prediction] for prediction in subtype_train_pred],
    }),
    pd.DataFrame({
        'record_id': record_ids[subtype_mask][subtype_test_idx],
        'true_label': [subtype_classes[index] for index in subtype_test_idx],
        'predicted_label': [subtype_label_names[prediction] for prediction in subtype_test_pred],
    }),
]
with pd.ExcelWriter(output_dir / 'abnormal_subtype_predictions.xlsx') as writer:
    subtype_predictions[0].to_excel(writer, sheet_name='training', index=False)
    subtype_predictions[1].to_excel(writer, sheet_name='testing', index=False)
with open(output_dir / 'abnormal_subtype_metrics.json', 'w', encoding='utf-8') as handle:
    json.dump(subtype_metrics, handle, indent=2, default=str)
with open(output_dir / 'two_stage_summary.json', 'w', encoding='utf-8') as handle:
    json.dump(
        {
            'handle_class_imbalance': HANDLE_CLASS_IMBALANCE,
            'max_minority_oversampling_ratio': MAX_MINORITY_OVERSAMPLING_RATIO,
            'beat_class_counts': beat_class_counts,
            'aami_filtered_partitions': filtered_aami_partitions,
            'stage1_total_sample_sizes': stage1_sample_sizes,
            'stage1_split_sample_sizes': stage1_split_sample_sizes,
            'stage2_total_sample_sizes': stage2_total_sample_sizes,
            'stage2_classification_sample_sizes': stage2_classification_sample_sizes,
            'stage2_classification_total': stage2_classification_total,
            'stage2_split_sample_sizes': stage2_split_sample_sizes,
            'binary': {
                'training': training_metrics,
                'test': testing_metrics,
            },
            'abnormal_subtypes': subtype_metrics,
        },
        handle,
        indent=2,
        default=str,
    )
log_message("Saved Stage 2 subtype metrics and predictions.")
stop_timer("overall")
log_timer_summary()

with log_path.open('w', encoding='utf-8') as handle:
    handle.write('\n'.join(log_messages) + '\n')


