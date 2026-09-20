"""Causal Shapley analysis utilities.

This module provides a lightweight wrapper around SHAP's KernelExplainer to
compute approximate Shapley values for the trained KNN classifier from the
Stigmergy pipeline. It uses background samples from training data to account
for feature dependence, which is a pragmatic approximation to conditional
expectations used in causal Shapley estimators.
"""
from typing import Optional
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_shap_values(model, X, background, nsamples=100, random_state=0):
    """Compute SHAP values using KernelExplainer.

    Args:
        model: A scikit-learn-like estimator with `predict_proba`.
        X: Samples to explain (2D array).
        background: Background samples for the explainer (2D array).
        nsamples: Number of evaluations per explanation (smaller => faster).
    Returns:
        shap_values: array of shape (n_samples, n_features) for the positive class.
        expected_value: base value from the explainer.
    """
    # KernelExplainer expects a function that returns model outputs for background
    f = lambda z: model.predict_proba(z)[:, 1]

    explainer = shap.KernelExplainer(f, background, link='identity')
    shap_values = explainer.shap_values(X, nsamples=nsamples, random_state=random_state)
    # KernelExplainer returns a list for binary classification; ensure array
    if isinstance(shap_values, list):
        shap_vals = np.array(shap_values)
        # For KernelExplainer, shap_values[1] typically corresponds to positive class
        shap_for_pos = shap_vals[1]
    else:
        shap_for_pos = np.array(shap_values)

    return shap_for_pos, explainer.expected_value


def plot_shap_summary(shap_values, feature_names, output_path=None, max_display=20):
    """Save a SHAP summary plot (bar) for the provided values.

    shap_values: array (n_samples, n_features)
    feature_names: list of feature names
    """
    arr = np.atleast_2d(shap_values)
    if arr.size == 0:
        mean_abs = np.zeros(len(feature_names), dtype=float)
    else:
        mean_abs = np.nanmean(np.abs(arr), axis=0)
    idx = np.argsort(mean_abs)[::-1][:max_display]
    names = [feature_names[i] for i in idx]
    values = mean_abs[idx]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.25)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color='#2b8cbe', edgecolor='none', height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP value|', fontsize=10)
    ax.set_title('SHAP feature importance', fontsize=11, pad=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.grid(axis='x', linestyle=':', color='#777777', alpha=0.3)
    fig.tight_layout()
    if output_path:
        fig.savefig(str(output_path), dpi=200)
        plt.close(fig)
    else:
        plt.show()


def plot_shap_beeswarm(shap_values, X=None, feature_names=None, output_path=None, max_display=20):
    """Save a SHAP beeswarm (dot) summary plot.

    shap_values: array (n_samples, n_features)
    X: optional feature matrix (n_samples, n_features) used for coloring
    """
    arr = np.atleast_2d(shap_values)
    if arr.size == 0:
        return
    # select top features by mean abs
    mean_abs = np.nanmean(np.abs(arr), axis=0)
    idx = np.argsort(mean_abs)[::-1][:max_display]
    sel_shap = arr[:, idx]
    sel_names = [feature_names[i] for i in idx] if feature_names is not None else None

    # shap.summary_plot expects shap_values in shape (n_samples, n_features)
    plt.figure(figsize=(8, max(4, len(sel_names) * 0.25)))
    try:
        # Use SHAP's summary_plot for beeswarm/dot
        shap.summary_plot(
            sel_shap,
            X[:, idx] if X is not None else None,
            feature_names=sel_names,
            show=False,
            color_bar=False,
            plot_size=0.6,
        )
        ax = plt.gca()
        ax.set_title('SHAP beeswarm', fontsize=11, pad=12)
        ax.set_xlabel('SHAP value', fontsize=10)
        ax.set_ylabel('Feature', fontsize=10)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        if output_path:
            plt.savefig(str(output_path), dpi=200, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    except Exception:
        plt.close()
        # Fallback: simple scatter per feature
        fig, ax = plt.subplots(figsize=(8, max(4, len(sel_names) * 0.25)))
        for i in range(sel_shap.shape[1]):
            y = sel_shap[:, i]
            x = np.random.normal(i, 0.05, size=len(y))
            ax.scatter(x, y, alpha=0.4, s=8, color='#2b8cbe', edgecolors='none')
        ax.set_xticks(range(len(sel_names)))
        ax.set_xticklabels(sel_names, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('SHAP value', fontsize=10)
        ax.set_xlabel('Feature', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        if output_path:
            fig.savefig(str(output_path), dpi=200, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()


def plot_shap_heatmap(shap_values, feature_names, output_path=None, n_samples=50, max_display=20):
    """Save a heatmap of SHAP values for the first `n_samples` and top features."""
    arr = np.atleast_2d(shap_values)
    if arr.size == 0:
        return
    mean_abs = np.nanmean(np.abs(arr), axis=0)
    idx = np.argsort(mean_abs)[::-1][:max_display]
    sel = arr[:n_samples, :][:, idx]
    sel_names = [feature_names[i] for i in idx]

    fig, ax = plt.subplots(figsize=(max(8, len(sel_names) * 0.25), max(4, min(n_samples, 50) * 0.08)))
    im = ax.imshow(sel, aspect='auto', cmap='RdBu_r')

    y_tick_step = max(1, sel.shape[0] // 10)
    y_ticks = list(range(0, sel.shape[0], y_tick_step))
    if sel.shape[0] - 1 not in y_ticks:
        y_ticks.append(sel.shape[0] - 1)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(i + 1) for i in y_ticks])

    x_tick_step = max(1, len(sel_names) // 20)
    x_ticks = list(range(0, len(sel_names), x_tick_step))
    if len(sel_names) - 1 not in x_ticks:
        x_ticks.append(len(sel_names) - 1)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([sel_names[i] for i in x_ticks], rotation=45, ha='right')

    ax.set_xlabel('Feature')
    ax.set_ylabel('Sample')
    cbar = fig.colorbar(im, ax=ax, label='SHAP value')
    cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    fig.tight_layout()
    if output_path:
        fig.savefig(str(output_path), dpi=200)
        plt.close(fig)
    else:
        plt.show()


def average_shap_by_group(shap_values, feature_names, groups):
    """Aggregate SHAP values by feature group mapping.

    groups: dict mapping group_name -> list of feature indices or names
    Returns: DataFrame with mean absolute SHAP per group.
    """
    arr = np.atleast_2d(shap_values)
    if arr.size == 0:
        mean_abs = np.zeros(len(feature_names), dtype=float)
    else:
        mean_abs = np.nanmean(np.abs(arr), axis=0)
    out = []
    for gname, cols in groups.items():
        if all(isinstance(c, int) for c in cols):
            vals = mean_abs[cols]
        else:
            idxs = [feature_names.index(c) for c in cols]
            vals = mean_abs[idxs]
        # use nanmean to be robust to empty selections
        mean_val = float(np.nanmean(vals)) if np.asarray(vals).size > 0 else 0.0
        out.append({'group': gname, 'mean_abs_shap': mean_val})
    return pd.DataFrame(out).sort_values('mean_abs_shap', ascending=False)
