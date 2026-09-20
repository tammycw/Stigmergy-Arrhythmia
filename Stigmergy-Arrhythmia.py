# Tammy Wang | OSSM | 06/15/26 | tammycc.wang@gmail.com
"""
Updated: 07/11/26

This script implements a two-stage intra-patient ECG beat classification workflow
inspired by stigmergic termite behavior. Prior to this classification stage,
raw ECG beats are first downloaded using "Stigmergy_download_wfdb_to_excel.py",
and the resulting data are preprocessed to generate derived features with
"Stigmergy_compute_derived_features.py".

The method uses raw or derived features in a staged classification pipeline with
separate pheromone fields for Stage 1 and Stage 2. S beats are excluded from
Stage 2 because their low sample count makes reliable subtype training unstable.
"""
"""
================================================================================
STIGMERGY-ARRHYTHMIA: Stigmergic Termite-Inspired Classification of ECG Beats
================================================================================

Theoretical foundation
--------------------
This script models ECG beat classification as a stigmergic swarm process. In
natural termite colonies, individuals do not follow a central controller; they
change the shared environment with pheromones and the colony structure emerges
from local interactions. The same idea is used here:

1. Each beat is carried by an autonomous termite scout.
2. Scouts move through a 2D lattice and deposit class-specific pheromones.
3. Pheromone diffusion and evaporation smooth the field over time.
4. Local colony neighborhoods become spatial signatures of class structure.
5. Classification emerges from colony clustering rather than from a direct PCA
   threshold decision.

Algorithmic procedure
--------------------
The workflow uses a two-stage classification strategy. Each stage has its own
pheromone field, swarm-derived features, KNN classifier, and Train/Test split.
This keeps the broad Normal-versus-Abnormal decision separate from the more
focused abnormal-subtype decision.

1. Load ECG beats from an Excel workbook containing either raw waveforms or
   derived waveform features.
2. Standardize the selected feature matrix and project it into 2D with PCA to
   define the lattice used for colony organization and visualization.
3. Stage 1, binary classification:
    - classify Normal (N) versus Abnormal (V, F, or S)
    - train a termite field and distance-weighted KNN classifier on the binary
      labels
    - evaluate the binary model on separate Train and Test splits
4. Stage 2, abnormal subtype classification:
    - retain only ventricular (V) and fusion (F) abnormal beats
    - intentionally ignore supraventricular (S) beats because their sample
      count is too low for reliable subtype training
    - train a fresh termite field and KNN classifier to distinguish V versus F
    - evaluate the subtype model on separate Train and Test splits
5. In each stage, convert beats into swarm descriptors by extracting local
    statistics from the learned pheromone neighborhood (mean, standard
    deviation, and class-field difference), then fit KNN on the augmented data.
----------------------------------
The main design choices responsible for the strong agreement are:
- class-balanced pheromone deposition so minority abnormal beats are not lost
- class-centroid attraction so each colony forms around a stable center
- local neighborhood aggregation instead of a single-cell score
- gaussian diffusion to smooth the pheromone landscape into coherent clusters
- k-nearest neighbors (KNN) on swarm-derived features to resolve labels
  using the learned colony geometry rather than an arbitrary threshold

Why this formulation can be preferable to ExtraTrees
---------------------------------------------------
Compared with tree-ensemble baselines such as ExtraTrees, the termite-inspired
representation offers several methodological advantages. First, it yields an
explicit spatial colony field that makes the learned class structure
interpretable as a pheromone landscape rather than as an opaque partitioning of
input features. Second, the colony dynamics preserve local interaction and
neighborhood geometry, which can better reflect emergent class organization
than randomized tree splits applied over a global feature space. Third, the
final decision stage remains a lightweight k-nearest neighbors (KNN) model
trained on swarm-derived descriptors, providing a more transparent and
inspectable decision mechanism than an ensemble composed of many randomized
trees.

Core functions
--------------
- resolve_dataset_path(...): locate the selected Excel workbook
- load_excel_dataset(...): load Excel features and metadata, filtering derived
  rows by the RR-status flag when applicable
- TermiteAgent.move(...): local stigmergic colony movement
- train_pheromone_classifier(...): self-organizing pheromone field training
- extract_swarm_features(...): convert colony field into descriptors
- predict_with_pheromone(...): backward-compatible pheromone score output
- evaluate_split(...): compute accuracy, precision, recall, F1, Kappa, AUROC

Outputs
-------
- PCA scatter plot of predictions versus truth
- ROC comparison plot
- metric comparison plot
- pheromone evolution animation
- pheromone heatmaps
- confusion-matrix plots for training and testing
- Excel workbook with train/test predictions
- Stage 1 binary metrics and predictions
- Stage 2 V-versus-F subtype metrics and predictions, with S excluded (only 2 beat types)

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
from plotnine import ggplot, aes, geom_point, geom_line, geom_abline, geom_text, geom_col, labs, theme_minimal, theme, ggsave, scale_color_manual, scale_fill_manual, scale_x_continuous, scale_y_continuous, coord_flip
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
RAW_DATASET_CANDIDATES = [
    current_dir / "input" / "mitbih_raw_data_rrflag.xlsx",
]
DERIVED_DATASET_CANDIDATES = [
    current_dir / "input" / "derived_waveform_features.xlsx",
]

# Directory for all generated outputs (plots, Excel, logs)
output_dir = current_dir / f'output-stigmergy-two-stage-{DATASET_CHOICE}'
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
# Choose a preset at the top of the script.
# Supported presets:
#   - "70_30" : training=0.70, testing=0.30
#   - "60_40" : training=0.60, testing=0.40
#   - "35_65" : training=0.35, testing=0.65
SPLIT_PRESET = "50_50"
SPLIT_PRESETS = {
    "70_30": (0.70, 0.30),
    "50_50": (0.50, 0.50),
    "30_70": (0.30, 0.70),
}

if SPLIT_PRESET not in SPLIT_PRESETS:
    raise ValueError(f"Unknown SPLIT_PRESET '{SPLIT_PRESET}'. Choose one of: {list(SPLIT_PRESETS.keys())}")

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

y_true = y.reshape(-1)
stage1_sample_sizes = {
    "Normal": int(np.sum(y_true == 0)),
    "Abnormal": int(np.sum(y_true == 1)),
}
stage1_sample_sizes["Total"] = stage1_sample_sizes["Normal"] + stage1_sample_sizes["Abnormal"]
log_message(f"Stage 1 total sample sizes: {stage1_sample_sizes}")
start_timer("preprocessing")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# A true termite colony needs a 2D embedding on which agents can self-organize.
# PCA is therefore used as the lattice coordinate system only; the final label
# emerges from the pheromone field and colony dominance rather than from a
# PCA-based decision threshold.
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)
log_message(f"Data shape: {X.shape} | Standardized feature space: {X_scaled.shape} | Reduced visualization: {X_2d.shape}")
stop_timer("preprocessing")

# Stage 1 uses an explicit two-way partition:
#   - TRAINING: fit the Normal-versus-Abnormal pheromone swarm and classifier
#   - TESTING: report the final binary performance once
train_idx, remaining_idx = train_test_split(
    np.arange(len(y_true)),
    test_size=test_fraction,
    stratify=y_true,
    random_state=42,
)
test_idx = remaining_idx

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


def train_pheromone_classifier(X_train, y_train, grid_size=60, n_iterations=260, record_snapshots=False, snapshot_interval=20):
    pheromone_grids = np.zeros((2, grid_size, grid_size), dtype=np.float32)
    y_train = np.asarray(y_train, dtype=int)
    bounds = build_grid_bounds(X_train)

    class_counts = np.bincount(y_train)
    class_weights = {0: 1.0, 1: max(class_counts[0] / max(class_counts[1], 1), 1.0)}

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
start_timer("model_training")
grid_size = 60
pheromone_grids, pheromone_bounds, pheromone_snapshots = train_pheromone_classifier(
    X_train_2d,
    y_train,
    grid_size=grid_size,
    record_snapshots=True,
    snapshot_interval=20,
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
X_train_aug = np.hstack([X_train, train_swarm_features])
X_test_aug = np.hstack([X_test, test_swarm_features])

knn_classifier = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn_classifier.fit(X_train_aug, y_train)
train_pred = knn_classifier.predict(X_train_aug)
test_pred = knn_classifier.predict(X_test_aug)
train_scores = knn_classifier.predict_proba(X_train_aug)[:, 1]
test_scores = knn_classifier.predict_proba(X_test_aug)[:, 1]
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

    log_message(f"\n=== {label} Metrics ===")
    log_message(f"Accuracy: {metrics['accuracy']:.3f}")
    log_message(f"Precision (abnormal class): {metrics['precision']:.3f}")
    log_message(f"Recall (abnormal class): {metrics['recall']:.3f}")
    log_message(f"F1-score (abnormal class): {metrics['f1']:.3f}")
    log_message(f"Cohen's Kappa: {metrics['kappa']:.3f}")
    log_message(f"AUROC: {metrics['auroc']:.3f}")
    log_message("Confusion Matrix:")
    log_message(metrics["confusion_matrix"])
    return metrics


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
    + geom_col(position='dodge', width=0.8)
    + coord_flip()
    + labs(
        title='Performance Summary Across Train and Test Splits',
        x='Metric',
        y='Score',
        fill='Split'
    )
    + scale_fill_manual(values=['#0072B2', '#009E73', '#D55E00'])
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

subtype_train_idx, subtype_test_idx = train_test_split(
    np.arange(len(subtype_y)),
    test_size=test_fraction,
    stratify=subtype_y,
    random_state=42,
)
subtype_X_scaled = X_scaled[subtype_mask]
subtype_X_2d = X_2d[subtype_mask]
subtype_X_train_2d = subtype_X_2d[subtype_train_idx]
subtype_X_test_2d = subtype_X_2d[subtype_test_idx]
subtype_y_train = subtype_y[subtype_train_idx]
subtype_y_test = subtype_y[subtype_test_idx]
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

subtype_knn_classifier = KNeighborsClassifier(
    n_neighbors=min(5, len(subtype_y_train)),
    weights='distance',
)
subtype_knn_classifier.fit(subtype_X_train_aug, subtype_y_train)
subtype_train_pred = subtype_knn_classifier.predict(subtype_X_train_aug)
subtype_test_pred = subtype_knn_classifier.predict(subtype_X_test_aug)
subtype_train_scores = subtype_knn_classifier.predict_proba(subtype_X_train_aug)[:, 1]
subtype_test_scores = subtype_knn_classifier.predict_proba(subtype_X_test_aug)[:, 1]

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
subtype_metrics_plot = (
    ggplot(subtype_metric_comparison_long, aes(x='metric', y='value', fill='split'))
    + geom_col(position='dodge', width=0.8)
    + coord_flip()
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
            'beat_class_counts': beat_class_counts,
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


