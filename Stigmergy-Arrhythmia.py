# Tammy Wang @ OSSM 06/15/26 tammycc.wang@gmail.com
"""
================================================================================
STIGMERGY-ARRHYTHMIA: Termite-Inspired Swarm Classification of ECG Beats
================================================================================

OVERVIEW:
This program implements a termite-inspired stigmergic algorithm to classify 
ECG beats as normal or abnormal using the MIT-BIH Arrhythmia Database.

================================================================================
THEORETICAL FOUNDATION - STIGMERGY & SWARM INTELLIGENCE:
================================================================================

Stigmergy is an emergent behavior mechanism where individual agents communicate 
indirectly through modification of their shared environment (pheromones), rather 
than through direct communication. This creates self-organizing collective behavior.

KEY CONCEPTS:
1. AUTONOMOUS AGENTS: Each termite agent operates independently with simple rules
2. LOCAL INTERACTIONS: Agents sense only their immediate neighborhood
3. PHEROMONE FIELD: Chemical markers left by agents guide future behavior
4. COLLECTIVE SEGREGATION: Despite no global coordination, agents naturally cluster
   like with like, creating spatial separation of beat classes

BIOLOGICAL INSPIRATION:
Real termites use pheromones to construct complex nests without a queen directing
construction. Our model adapts this for ECG classification:
- Normal beats (class 0) deposit pheromones in their local regions
- Abnormal beats (class 1) deposit pheromones differently weighted
- Over iterations, a spatial pheromone map emerges that separates the classes
- New beats are classified based on which pheromone field is stronger

MECHANISM:
1. Initialize agents at grid positions corresponding to their ECG features
2. Each iteration: agents move randomly, deposit pheromones of their class
3. Pheromones diffuse (smooth) and evaporate (decay) - simulating natural diffusion
4. After convergence: pheromone maps encode class boundaries
5. New samples are classified by comparing pheromone strengths at their location

================================================================================
KEY FUNCTIONS & PROCEDURES:
================================================================================

1. DATA LOADING & PREPROCESSING:
   - load_mit_bih_data(record_numbers, samples_per_beat, max_samples)
     Load ECG records from PhysioNet MIT-BIH database
   - load_from_excel(path)
     Load pre-cached ECG data from Excel file
   - StandardScaler, PCA: Normalize and reduce data to 2D for visualization

2. SWARM AGENT SIMULATION:
   - class TermiteAgent: Represents individual termite agents carrying beat data
   - TermiteAgent.move(grid_size): Simulates random walk movement on grid
   - scale_to_grid(point, grid_size): Maps 2D ECG points to grid coordinates

3. PHEROMONE-BASED CLASSIFIER:
   - train_pheromone_classifier(X_train, y_train, grid_size, n_iterations)
     Simulates termite swarm to deposit pheromones separating beat classes
   - predict_with_pheromone(X_eval, pheromone_grids, grid_size)
     Classifies new beats based on pheromone concentrations

4. EVALUATION & METRICS:
   - evaluate_split(y_true, y_pred, scores, label)
     Computes accuracy, precision, recall, F1, Kappa, and AUROC

5. VISUALIZATION & OUTPUT:
   - Scatter plot: Test set predictions vs. actual labels (PCA space)
   - ROC curve: Training vs. Testing AUROC comparison
   - Metrics comparison: Bar chart of all performance metrics
   - Excel export: Training and testing predictions
   - Log file: Summary of all operations and metrics

WORKFLOW:
   1. Load data (MIT-BIH or Excel cache)
   2. Preprocess: standardize & reduce to 2D via PCA
   3. Train/test split (70/30)
   4. Train termite swarm pheromone classifier
   5. Evaluate on both train and test sets
   6. Generate visualizations and save results

================================================================================
"""

import os
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score, roc_auc_score, roc_curve
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from plotnine import ggplot, aes, geom_point, geom_line, geom_abline, geom_text, geom_col, labs, theme_minimal, theme, ggsave, scale_color_manual, scale_fill_manual, scale_x_continuous, scale_y_continuous, coord_flip
from plotnine.themes.elements import element_blank, element_line, element_rect, element_text


current_dir = Path(__file__).resolve().parent
log_path = current_dir / "stigmergy_run.log"
log_messages = []


def log_message(message):
    text = str(message)
    print(text)
    log_messages.append(text)


log_message(f"Current directory: {current_dir}")

# Choose data source:
# - "excel" : read cached data from mitbih_data.xlsx
# - "database" : reload data from the MIT-BIH source (PhysioNet)
# data_source = "database"  # extract data from MIT-BIH 
data_source = "excel"  # change to "database" to reload from MIT-BIH
max_samples = 12000
excel_path = current_dir / "mitbih_data.xlsx"

# ====================== 1. Load MIT-BIH Data ======================
def load_mit_bih_data(record_numbers=[100, 101, 102, 103, 105, 106, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 220, 221, 222, 223, 228, 230, 231, 232, 233, 234], samples_per_beat=180, max_samples=max_samples):
    X = []
    y = []
    record_ids = []
    beat_classes = []
    
    for rec in record_numbers:
        try:
            # Download from PhysioNet if not present
            record = wfdb.rdrecord(str(rec), pn_dir='mitdb')
            annotation = wfdb.rdann(str(rec), 'atr', pn_dir='mitdb')
            
            signal = record.p_signal[:, 0]  # Use MLII lead (common)
            
            for i, symbol in enumerate(annotation.symbol):
                if symbol in ['N', 'V', 'S', 'F']:  # Focus on main classes: Normal (N), Ventricular Ectopic (V), Supraventricular (S), Fusion (F)
                    center = annotation.sample[i]
                    half = samples_per_beat // 2
                    
                    if center - half >= 0 and center + half < len(signal):
                        segment = signal[center - half:center + half]
                        X.append(segment)
                        # Map to classes: 0=Normal, 1=Abnormal
                        label = 0 if symbol == 'N' else 1
                        y.append(label)
                        record_ids.append(rec)
                        beat_classes.append(symbol)

                        if max_samples is not None and len(X) >= max_samples:
                            break
            if max_samples is not None and len(X) >= max_samples:
                break
        except Exception as e:
            print(f"Error loading record {rec}: {e}")
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    record_ids = np.array(record_ids, dtype=int)
    beat_classes = np.array(beat_classes, dtype=object)
    return X, y, record_ids, beat_classes


def load_from_excel(path):
    path = Path(path)
    features = pd.read_excel(path, sheet_name="features")
    labels = pd.read_excel(path, sheet_name="labels")
    metadata = pd.read_excel(path, sheet_name="metadata")

    X = features.to_numpy(dtype=np.float32)
    y = labels.iloc[:, 0].to_numpy(dtype=np.float32).reshape(-1, 1)
    record_ids = metadata["record_id"].to_numpy(dtype=int)
    beat_classes = metadata["beat_class"].to_numpy(dtype=object)
    return X, y, record_ids, beat_classes


log_message("Loading MIT-BIH data...")
if data_source == "excel" and excel_path.exists():
    X, y, record_ids, beat_classes = load_from_excel(excel_path)
    log_message(f"Loaded data from {excel_path}")
else:
    X, y, record_ids, beat_classes = load_mit_bih_data(max_samples=max_samples)  # Larger subset for better training coverage
    log_message("Loaded data from MIT-BIH source")

    # Save the loaded data for reuse as an Excel file
    feature_df = pd.DataFrame(X, dtype=float)
    label_df = pd.DataFrame(y.reshape(-1, 1), columns=["label"])
    metadata_df = pd.DataFrame({
        "record_id": record_ids,
        "beat_class": beat_classes,
        "label": y.reshape(-1),
    })
    with pd.ExcelWriter(excel_path) as writer:
        feature_df.to_excel(writer, sheet_name="features", index=False)
        label_df.to_excel(writer, sheet_name="labels", index=False)
        metadata_df.to_excel(writer, sheet_name="metadata", index=False)
    log_message(f"Saved MIT-BIH data to {excel_path}")

y_true = y.reshape(-1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce to 2D for visualization & agent grid
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)
log_message(f"Data shape: {X.shape} | Reduced: {X_2d.shape}")

# Split into train and test sets so the evaluation reflects real predictive performance.
X_train, X_test, y_train, y_test = train_test_split(
    X_2d,
    y_true,
    test_size=0.3,
    stratify=y_true,
    random_state=42,
)

# ====================== 2. Termite-Inspired Swarm Model ======================

class TermiteAgent:
    """
    Represents a single termite agent carrying an ECG beat sample.
    
    Attributes:
        pos (np.ndarray): Current grid position [x, y] (changes via random walk)
        carrying (np.ndarray): The ECG beat sample data this agent represents
        class_label (int): Class of the beat (0=Normal, 1=Abnormal)
        pheromone_memory (float): Memory of pheromone field at current location
                                  (reserved for future use with directed movement)
    """
    
    def __init__(self, position, sample_point, class_label):
        self.pos = position  # (x, y) in 2D space on discretized grid
        self.carrying = sample_point  # Original feature vector from PCA
        self.class_label = int(class_label)  # 0=Normal, 1=Abnormal
        self.pheromone_memory = 0.0  # Can track local pheromone concentration

    def move(self, grid_size=50):
        """
        Execute a random walk step on the grid (Brownian motion approximation).
        
        MECHANISM: Each agent takes a small random step in 2D space:
        - dx, dy ∈ {-1, 0, 1} with uniform probability
        - Positions are clipped to stay within [0, grid_size-1]
        
        This simple diffusion allows agents to explore the space and gradually
        settle into regions where similar class members congregate.
        
        Args:
            grid_size (int): Maximum grid dimension (agents bounded to this)
        """
        # Random step: uniform choice from {-1, 0, +1} for each axis
        dx = np.random.randint(-1, 2)
        dy = np.random.randint(-1, 2)
        # Update position with boundary clipping
        self.pos = np.clip(self.pos + [dx, dy], 0, grid_size-1)



def scale_to_grid(point, grid_size):
    """
    PROJECT ECG FEATURES TO DISCRETE GRID:
    
    Transform continuous 2D ECG features (from PCA) to discrete grid coordinates.
    This is necessary because:
    1. Pheromone grids are discrete 2D arrays (sparse representation)
    2. Multiple beats near the same location reinforce pheromone at that grid cell
    3. Aggregation creates sharp class boundaries
    
    MAPPING STRATEGY (Min-Max Normalization):
    - Find data bounds: [x_min, x_max] × [y_min, y_max] from training data
    - Normalize point to [0, 1]: (point - min) / (max - min)
    - Scale to grid indices: normalized_value * (grid_size - 1)
    - Clip to valid range [0, grid_size - 1]
    
    INTUITION: Beats in high-density regions (core of each class) will have many
    agents depositing pheromones there, creating peaks. Sparse boundary regions
    will have weaker pheromones, allowing classification via pheromone gradient.
    
    Args:
        point (np.ndarray): Single 2D ECG feature vector [x, y]
        grid_size (int): Resolution of discretized grid (e.g., 60×60)
    
    Returns:
        np.ndarray: Grid coordinates [x_idx, y_idx] as float (for agent position)
    """
    # Extract training data bounds in each dimension
    x_min, x_max = X_train[:, 0].min(), X_train[:, 0].max()
    y_min, y_max = X_train[:, 1].min(), X_train[:, 1].max()

    # Compute span; use 1e-8 to avoid division by zero (if all points identical)
    x_span = max(x_max - x_min, 1e-8)
    y_span = max(y_max - y_min, 1e-8)

    # Normalize to [0, grid_size-1] and clip to bounds
    x_idx = int(np.clip((point[0] - x_min) / x_span * (grid_size - 1), 0, grid_size - 1))
    y_idx = int(np.clip((point[1] - y_min) / y_span * (grid_size - 1), 0, grid_size - 1))
    return np.array([x_idx, y_idx], dtype=float)


def train_pheromone_classifier(X_train, y_train, grid_size=60, n_iterations=220, record_snapshots=False, snapshot_interval=20):
    pheromone_grids = np.zeros((2, grid_size, grid_size), dtype=np.float32)
    y_train = np.asarray(y_train, dtype=int)
    class_counts = np.bincount(y_train)
    class_weights = {0: 1.0, 1: max(class_counts[0] / max(class_counts[1], 1), 1.0)}

    agents = [
        TermiteAgent(scale_to_grid(point, grid_size), point, label)
        for point, label in zip(X_train, y_train)
    ]

    snapshots = []
    if record_snapshots:
        snapshots.append((0, pheromone_grids.copy()))

    log_message("Running termite-inspired stigmergy training...")
    for it in range(n_iterations):
        for agent in agents:
            agent.move(grid_size)
            x_idx, y_idx = agent.pos.astype(int)
            deposit_strength = 0.6 * class_weights[agent.class_label]
            pheromone_grids[agent.class_label, x_idx, y_idx] += deposit_strength

        pheromone_grids[0] = gaussian_filter(pheromone_grids[0], sigma=0.8)
        pheromone_grids[1] = gaussian_filter(pheromone_grids[1], sigma=0.8)
        pheromone_grids *= 0.995

        if record_snapshots and (it + 1) % snapshot_interval == 0:
            snapshots.append((it + 1, pheromone_grids.copy()))

        if it % 50 == 0:
            log_message(f"Iter {it:3d} | Normal pheromone: {pheromone_grids[0].mean():.4f} | Abnormal pheromone: {pheromone_grids[1].mean():.4f}")

    for cls in range(2):
        max_val = np.max(pheromone_grids[cls])
        if max_val > 0:
            pheromone_grids[cls] = pheromone_grids[cls] / max_val

    if record_snapshots:
        return pheromone_grids, snapshots
    return pheromone_grids


def predict_with_pheromone(X_eval, pheromone_grids, grid_size=60):
    predicted = []
    score_diffs = []
    for point in X_eval:
        pos = scale_to_grid(point, grid_size)
        x_idx, y_idx = int(pos[0]), int(pos[1])
        normal_score = pheromone_grids[0, x_idx, y_idx]
        abnormal_score = pheromone_grids[1, x_idx, y_idx]
        score_diff = abnormal_score - normal_score
        score_diffs.append(score_diff)
        predicted.append(1 if score_diff > 0.05 else 0)
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


# ====================== 3. Train and evaluate ======================
grid_size = 60
pheromone_grids, pheromone_snapshots = train_pheromone_classifier(
    X_train,
    y_train,
    grid_size=grid_size,
    record_snapshots=True,
    snapshot_interval=20,
)

train_pred, train_scores = predict_with_pheromone(X_train, pheromone_grids, grid_size=grid_size)
test_pred, test_scores = predict_with_pheromone(X_test, pheromone_grids, grid_size=grid_size)

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

metric_comparison_df = pd.DataFrame({
    'metric': ['Accuracy', 'Precision', 'Recall', 'F1', 'Kappa', 'AUROC'],
    'Training': [training_metrics['accuracy'], training_metrics['precision'], training_metrics['recall'], training_metrics['f1'], training_metrics['kappa'], training_metrics['auroc']],
    'Testing': [testing_metrics['accuracy'], testing_metrics['precision'], testing_metrics['recall'], testing_metrics['f1'], testing_metrics['kappa'], testing_metrics['auroc']],
})
metric_comparison_long = pd.melt(
    metric_comparison_df,
    id_vars=['metric'],
    value_vars=['Training', 'Testing'],
    var_name='split',
    value_name='value'
)
metrics_plot = (
    ggplot(metric_comparison_long, aes(x='metric', y='value', fill='split'))
    + geom_col(position='dodge', width=0.8)
    + coord_flip()
    + labs(
        title='Training vs Testing Metrics Comparison',
        x='Metric',
        y='Score',
        fill='Split'
    )
    + scale_fill_manual(values=['#1f77b4', '#d62728'])
    + scale_y_continuous(limits=(0, 1.0), expand=(0, 0))
    + theme_minimal(base_size=13)
    + theme(
        plot_title=element_text(hjust=0.5, size=16, weight='bold'),
        legend_position='bottom',
        panel_grid_major_y=element_blank(),
        panel_grid_minor=element_blank(),
        axis_line=element_line(color='black'),
        panel_border=element_rect(color='black', fill=None, size=1),
        axis_ticks=element_line(color='black', size=0.6)
    )
)

# Visualization with ggplot2-style grammar of graphics
viz_df = pd.DataFrame({
    'x': X_test[:, 0],
    'y': X_test[:, 1],
    'actual': np.where(y_test == 0, 'Normal', 'Abnormal'),
    'predicted': np.where(test_pred == 0, 'Normal', 'Abnormal')
})

plot = (
    ggplot(viz_df, aes(x='x', y='y', color='predicted', shape='actual'))
    + geom_point(alpha=0.7, size=2.5)
    + labs(
        title='Binary Termite Swarm Classification of MIT-BIH ECG Beats',
        x='PCA Component 1',
        y='PCA Component 2',
        color='Predicted Class',
        shape='Actual Class'
    )
    + theme_minimal()
)

fpr_train, tpr_train, _ = roc_curve(y_train, train_scores)
fpr_test, tpr_test, _ = roc_curve(y_test, test_scores)
roc_df = pd.DataFrame({
    'fpr': np.concatenate([fpr_train, fpr_test]),
    'tpr': np.concatenate([tpr_train, tpr_test]),
    'split': np.concatenate([np.repeat('Training', len(fpr_train)), np.repeat('Testing', len(fpr_test))]),
    'auroc': np.concatenate([
        np.repeat(training_metrics['auroc'], len(fpr_train)),
        np.repeat(testing_metrics['auroc'], len(fpr_test))
    ])
})
annotation_df = pd.DataFrame({
    'split': ['Training', 'Testing'],
    'x': [0.72, 0.72],
    'y': [0.18, 0.10],
    'label': [f'Training AUROC = {training_metrics["auroc"]:.2f}', f'Testing AUROC = {testing_metrics["auroc"]:.2f}']
})
roc_plot = (
    ggplot(roc_df, aes(x='fpr', y='tpr', color='split'))
    + geom_line(size=1.2)
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
        title='Receiver Operating Characteristic (ROC) Curves',
        x='False Positive Rate',
        y='True Positive Rate',
        color='Split'
    )
    + scale_color_manual(values=['#1f77b4', '#d62728'])
    + scale_x_continuous(limits=(0, 1), expand=(0, 0))
    + scale_y_continuous(limits=(0, 1), expand=(0, 0))
    + theme_minimal(base_size=13)
    + theme(
        plot_title=element_text(hjust=0.5, size=16, weight='bold'),
        legend_position='bottom',
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        axis_line=element_line(color='black'),
        panel_border=element_rect(color='black', fill=None, size=1),
        axis_ticks=element_line(color='black', size=0.6)
    )
)

output_dir = current_dir
output_dir.mkdir(parents=True, exist_ok=True)
pdf_path = output_dir / 'stigmergy_clustering.pdf'
roc_pdf_path = output_dir / 'stigmergy_auroc.pdf'
metrics_pdf_path = output_dir / 'stigmergy_metrics_comparison.pdf'
process_gif_path = output_dir / 'stigmergy_pheromone_evolution.gif'
pheromone_heatmap_path = output_dir / 'stigmergy_pheromone_heatmap.pdf'
prediction_path = output_dir / 'stigmergy_predictions.xlsx'
plot.save(filename=str(pdf_path), width=10, height=8, dpi=300)
roc_plot.save(filename=str(roc_pdf_path), width=8, height=6, dpi=300)
metrics_plot.save(filename=str(metrics_pdf_path), width=8, height=6, dpi=300)
save_pheromone_animation(pheromone_snapshots, process_gif_path)
save_pheromone_heatmaps(pheromone_grids, pheromone_heatmap_path)

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
log_message(f'Saved predictions to {prediction_path}')

with log_path.open('w', encoding='utf-8') as handle:
    handle.write('\n'.join(log_messages) + '\n')