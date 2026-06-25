# Stigmergy-Arrhythmia: Swarm Intelligence for Arrhythmia Classification

A bio-inspired machine learning classifier that uses **stigmergic algorithms** (termite swarm behavior) to distinguish normal from abnormal ECG heartbeats.

## Overview

This project implements a novel approach to arrhythmia classification inspired by the collective behavior of termites. Rather than using traditional supervised learning methods, we simulate a swarm of autonomous agents that communicate through pheromone deposition—a process called *stigmergy*. Over multiple iterations, the agents self-organize to create decision boundaries that separate normal and abnormal cardiac beats in a low-dimensional feature space.

## Theory: Stigmergy & Swarm Intelligence

**Stigmergy** is an emergent behavior mechanism where individual agents communicate **indirectly** through modification of their shared environment (pheromones), rather than through direct communication. This creates self-organizing collective behavior with no central coordinator.

### Key Concepts
- **Autonomous Agents**: Each termite agent operates independently with simple rules
- **Local Interactions**: Agents sense only their immediate neighborhood
- **Pheromone Field**: Chemical markers left by agents guide future behavior
- **Collective Segregation**: Despite no global coordination, agents naturally cluster like with like

### Biological Inspiration
Real termites use pheromones to construct complex nests without a queen directing construction. Our model adapts this for ECG classification:
- Normal beats (class 0) deposit pheromones in their local regions
- Abnormal beats (class 1) deposit pheromones differently weighted (class imbalance correction)
- Over iterations, a spatial pheromone map emerges that separates the classes
- New beats are classified based on which pheromone field is stronger

### Algorithm Mechanism
1. **Initialization**: Place agents at grid positions corresponding to their 2D ECG features
2. **Iteration**: Each timestep, agents move randomly and deposit pheromones of their class
3. **Diffusion**: Gaussian smoothing simulates pheromone diffusion across the grid
4. **Decay**: Pheromones evaporate over time (multiplicative decay factor)
5. **Convergence**: After 220 iterations, pheromone maps encode class boundaries
6. **Classification**: New samples are classified by comparing pheromone strengths at their location

## Features

**Swarm-based Classification**  
Autonomous termite agents explore feature space and self-organize into decision boundaries

**Real Cardiology Data**  
Processes MIT-BIH Arrhythmia Database from PhysioNet (46 patient records, 12,000+ beats)

**Dimensionality Reduction**  
PCA projects 180-dimensional ECG signals to 2D for visualization and computation

**Comprehensive Evaluation**  
- Accuracy, Precision, Recall, F1-score
- Cohen's Kappa (inter-rater agreement)
- AUROC with ROC curves for train/test splits
- Confusion matrix analysis

**Publication-Ready Visualizations**  
- Scatter plots of predicted vs. actual classes in PCA space
- ROC curves with AUROC annotations
- Metrics comparison bar charts
- All plots generated with plotnine (ggplot2-style grammar of graphics)

**Data Caching**  
First run downloads from PhysioNet; subsequent runs use cached Excel file for speed

**Detailed Logging**  
Session log records all operations, iterations, and metrics

## Installation

### Requirements
- Python 3.8+
- scikit-learn
- pandas
- numpy
- scipy
- plotnine
- wfdb (for PhysioNet data access)

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/Stigmergy-Arrhythmia.git
cd Stigmergy-Arrhythmia

# Install dependencies
pip install -r requirements.txt
```

### Requirements File
Create `requirements.txt`:
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=0.24.0
scipy>=1.7.0
wfdb>=3.3.0
plotnine>=0.9.0
```

## Usage

### Basic Run
```bash
python Stigmergy-Arrhythmia.py
```

### Configuration
Edit the following parameters in the script:

```python
# Data source: "database" or "excel"
data_source = "database"  # First run: download from MIT-BIH
# data_source = "excel"   # Subsequent runs: use cached data

# Maximum samples to load
max_samples = 12000

# Grid resolution for swarm simulation
grid_size = 60

# Number of training iterations
n_iterations = 220
```

## Outputs

The script generates:

1. **stigmergy_clustering.pdf**  
   Scatter plot of test set with predicted vs. actual labels in PCA space

2. **stigmergy_auroc.pdf**  
   ROC curves comparing training and testing AUROC

3. **stigmergy_metrics_comparison.pdf**  
   Bar chart comparing all metrics across train/test splits

4. **stigmergy_predictions.xlsx**  
   Excel file with training and testing predictions for further analysis

5. **stigmergy_run.log**  
   Detailed log including:
   - Data loading summary
   - Swarm iteration progress
   - All performance metrics
   - File save confirmations


## Project Structure

```
Stigmergy-Arrhythmia/
├── Stigmergy-Arrhythmia.py       # Main script
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── mitbih_data.xlsx              # Cached data (auto-generated)
├── stigmergy_clustering.pdf       # Output: classification plot
├── stigmergy_auroc.pdf           # Output: ROC curves
├── stigmergy_metrics_comparison.pdf # Output: metrics comparison
├── stigmergy_predictions.xlsx    # Output: predictions
└── stigmergy_run.log             # Output: session log
```

## How the Swarm Works: Step-by-Step

### 1. **Agent Initialization**
```
For each ECG beat in training data:
  - Create a TermiteAgent
  - Position on grid based on 2D PCA coordinates
  - Assign class label (0=Normal, 1=Abnormal)
```

### 2. **Random Walk Movement**
```
Each iteration:
  - Each agent moves randomly: dx, dy ∈ {-1, 0, +1}
  - Position clipped to grid boundaries [0, grid_size-1]
```

### 3. **Pheromone Deposition**
```
At each position visited:
  - Deposit pheromone strength = 0.6 × class_weight
  - class_weight handles imbalanced datasets
  - Normal class weight = 1.0
  - Abnormal class weight = (count_normal / count_abnormal)
```

### 4. **Pheromone Diffusion & Decay**
```
After all agents move:
  - Apply Gaussian filter (σ=0.8) to simulate diffusion
  - Multiply by decay factor (0.995) to simulate evaporation
  - Result: sharp peaks at class centers, gradual falloff
```

### 5. **Classification**
```
For each test sample at position P:
  - Query normal pheromone at P: φ_normal(P)
  - Query abnormal pheromone at P: φ_abnormal(P)
  - Score difference: Δ = φ_abnormal(P) - φ_normal(P)
  - Classify as abnormal if Δ > 0.05, else normal
```

## Key Implementation Details

- **Grid Size**: 60×60 provides good resolution for the MIT-BIH data
- **Iterations**: 220 iterations balances convergence with computation time
- **Gaussian Sigma**: 0.8 provides smooth diffusion without over-blurring
- **Decay Factor**: 0.995 per iteration gives stable pheromone landscapes
- **Class Imbalance Handling**: Weighted deposition accounts for unequal class sizes
- **Normalization**: MinMax scaling on test data during projection ensures consistent grid mapping

## Scientific Background

This work is inspired by:
- **Termite Mound Construction**: Décamps & Theraulaz (1990) on stigmergic coordination
- **Swarm Robotics**: Dorigo & Birattari (2007) on collective intelligence
- **Ant Colony Optimization**: Dorigo et al. (2006) for combinatorial optimization
- **Self-Organizing Maps**: Kohonen (1990) for unsupervised clustering

## Limitations & Future Work

### Current Limitations
- Binary classification only (normal vs. abnormal)
- Grid discretization may lose fine feature details
- Performance depends on PCA to 2D (may not capture all information)
- Random walk is simple; could benefit from directed movement based on pheromone gradients

### Future Enhancements
- Multi-class extension (normal, VEB, SVB, fusion beats)
- Adaptive pheromone deposition based on classification confidence
- 3D or higher-dimensional agent space
- Comparison with standard ML baselines (SVM, Random Forest, Neural Networks)
- Temporal information (beat-to-beat intervals) integration
- Real-time prediction on streaming ECG data

## Citation

If you use this project in research, please cite:

```bibtex
@misc{wang2026stigmergy,
  title={Stigmergy-Arrhythmia: Swarm Intelligence for ECG Beat Classification},
  author={Wang, Tammy},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/yourusername/Stigmergy-Arrhythmia}}
}
```

## License

MIT License - See LICENSE file for details

## Author

**Tammy Wang**  
Email: tammycc.wang@gmail.com  
Institution: OSSM (Oklahoma School of Science and Mathematics)  
Date: June 2026

## Acknowledgments

- MIT-BIH Arrhythmia Database from PhysioNet
- scikit-learn and pandas communities
- plotnine developers for ggplot2-style visualization in Python

---

**Questions or Contributions?** Feel free to open an issue or submit a pull request!
