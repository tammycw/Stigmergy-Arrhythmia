# Tammy Wang | OSSM | 07/11/26 | tammycc.wang@gmail.com

"""Compute derived waveform features from ECG beat segments and export them to Excel.

This script starts from beat-level ECG segment rows in an Excel workbook and
adds a richer set of handcrafted features for downstream modeling. The feature
families include:

- waveform statistics: mean, spread, range, percentiles, energy, RMS, zero
  crossings, slope, and area;
- QRS characterization: peak position, amplitude, duration, and the pre-/post-
  peak slopes around the dominant beat peak;
- spectral descriptors: centroid, bandwidth, dominant frequency, and power in
  fixed frequency bands;
- nonlinear and complexity measures: sample entropy and Hjorth-based activity,
  mobility, and complexity;
- wavelet features: multiscale energy content from discrete wavelet
  decomposition;
- RR intervals: previous and next beat-to-beat intervals derived from sample
  positions within each recording.

The resulting features are written to a workbook with separate sheets for
feature values, metadata, and feature definitions.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch
import pywt

current_dir = Path(__file__).resolve().parent
# Write derived features to the input directory so they can be reused in later
# analysis steps. The raw input data is still required for this stage.
output_dir = current_dir / 'input'
# output_dir.mkdir(parents=True, exist_ok=True)


def _print_progress(current, total, prefix='Progress'):
    if total <= 0:
        return
    done = int(20 * current / total)
    bar = '#' * done + '-' * (20 - done)
    print(f"\r{prefix}: [{bar}] {current}/{total}", end='', flush=True)


def _finish_progress():
    print()


def _sample_entropy(x, m=2, r=None):
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N < 2 * (m + 1):
        return np.nan
    if r is None:
        r = 0.2 * np.std(x)

    def _phi(window_len):
        n_windows = N - window_len + 1
        if n_windows < 2:
            return 0

        windows = np.array([x[i:i + window_len] for i in range(n_windows)], dtype=float)
        diff = np.abs(windows[:, None, :] - windows[None, :, :]).max(axis=2)
        matches = diff <= r
        return int(np.count_nonzero(np.triu(matches, 1)))

    try:
        B = _phi(m)
        A = _phi(m + 1)
        if B == 0:
            return np.nan
        return -np.log(A / B) if A != 0 else np.inf
    except Exception:
        return np.nan


def compute_rr_intervals(sample_positions, record_ids, fs=360.0):
    rr_prev = np.full(len(record_ids), np.nan, dtype=float)
    rr_next = np.full(len(record_ids), np.nan, dtype=float)
    rr_problem_flag = np.full(len(record_ids), 'ok', dtype=object)

    if sample_positions is None or len(sample_positions) != len(record_ids):
        return rr_prev, rr_next, rr_problem_flag

    sample_positions = np.asarray(sample_positions, dtype=float)
    record_ids = np.asarray(record_ids)

    unique_records = np.unique(record_ids)
    print(f"Computing RR intervals for {len(unique_records)} record(s)...")
    for idx, record_id in enumerate(unique_records, start=1):
        _print_progress(idx, len(unique_records), prefix='RR records')
        mask = record_ids == record_id
        rec_positions = sample_positions[mask]
        finite_mask = np.isfinite(rec_positions)
        valid_positions = rec_positions[finite_mask]
        record_idx = np.flatnonzero(mask)[finite_mask]

        if valid_positions.size < 2:
            rr_problem_flag[mask] = f"excluded_rr_unusable: only {valid_positions.size} valid beat position(s)"
            print(f"Excluding record {record_id} from RR features: only {valid_positions.size} valid beat position(s)")
            continue

        sort_idx = np.argsort(valid_positions)
        sorted_positions = valid_positions[sort_idx]
        sorted_idx = record_idx[sort_idx]

        rr_series = np.diff(sorted_positions) / float(fs)
        plausible_mask = np.isfinite(rr_series) & (rr_series >= 0.2) & (rr_series <= 2.5)
        plausible_count = np.count_nonzero(plausible_mask)
        min_required = max(1, int(np.ceil(rr_series.size / 2.0)))

        rr_prev_values = np.full(sorted_positions.size, np.nan, dtype=float)
        rr_next_values = np.full(sorted_positions.size, np.nan, dtype=float)
        if rr_series.size > 0:
            rr_prev_values[1:] = rr_series
            rr_next_values[:-1] = rr_series

        rr_prev[sorted_idx] = rr_prev_values
        rr_next[sorted_idx] = rr_next_values

        if plausible_count < min_required:
            rr_problem_flag[mask] = f"excluded_rr_unusable: only {plausible_count}/{rr_series.size} plausible RR intervals"
            print(
                f"Record {record_id} has implausible RR intervals: only {plausible_count}/{rr_series.size} plausible RR intervals"
            )

        if sorted_idx.size:
            rr_problem_flag[sorted_idx[0]] = 'edge_first_beat'
            rr_problem_flag[sorted_idx[-1]] = 'edge_last_beat'

    _finish_progress()
    return rr_prev, rr_next, rr_problem_flag


def compute_waveform_features(X_segments, fs=360.0, rr_prev=None, rr_next=None):
    X_segments = np.asarray(X_segments, dtype=float)
    if X_segments.ndim == 1:
        X_segments = X_segments.reshape(1, -1)

    n_samples, L = X_segments.shape
    print(f"Computing waveform features for {n_samples} beat(s) with length {L}...")
    mean = X_segments.mean(axis=1)
    std = X_segments.std(axis=1)
    ptp = np.ptp(X_segments, axis=1)
    mx = X_segments.max(axis=1)
    mn = X_segments.min(axis=1)
    med = np.median(X_segments, axis=1)
    p25 = np.percentile(X_segments, 25, axis=1)
    p75 = np.percentile(X_segments, 75, axis=1)
    energy = np.sum(X_segments ** 2, axis=1)
    rms = np.sqrt(np.mean(X_segments ** 2, axis=1))
    zero_cross = np.sum((X_segments[:, :-1] * X_segments[:, 1:]) < 0, axis=1)
    max_slope = np.max(np.abs(np.diff(X_segments, axis=1)), axis=1)
    area = np.sum(X_segments, axis=1) / float(fs)

    r_idx = np.argmax(np.abs(X_segments), axis=1)
    r_idx_rel = r_idx / float(L)
    qrs_amp = np.abs(X_segments[np.arange(n_samples), r_idx]) - np.median(X_segments, axis=1)
    qrs_duration = np.zeros(n_samples, dtype=float)
    pre_slope = np.zeros(n_samples, dtype=float)
    post_slope = np.zeros(n_samples, dtype=float)
    corr_template = np.full(n_samples, np.nan, dtype=float)

    template = np.nanmedian(X_segments, axis=0)

    sk = skew(X_segments, axis=1, nan_policy='omit')
    kt = kurtosis(X_segments, axis=1, nan_policy='omit')

    centroid = np.full(n_samples, np.nan, dtype=float)
    bandwidth = np.full(n_samples, np.nan, dtype=float)
    dom_freq = np.full(n_samples, np.nan, dtype=float)
    band_0_5 = np.zeros(n_samples, dtype=float)
    band_5_15 = np.zeros(n_samples, dtype=float)
    band_15_40 = np.zeros(n_samples, dtype=float)

    # per-sample computations combined into one loop so progress can be shown
    sampen_vals = np.full(n_samples, np.nan, dtype=float)
    try:
        wavelet = 'db4'
        max_level = 3
        wave_energies = np.zeros((n_samples, max_level), dtype=float)
    except Exception:
        wavelet = 'db4'
        max_level = 3
        wave_energies = np.zeros((n_samples, 3), dtype=float)

    for i in range(n_samples):
        x = X_segments[i]
        # QRS-derived features
        try:
            pk = int(r_idx[i])
            half_h = 0.5 * np.abs(x[pk])
            above = np.where(np.abs(x) >= half_h)[0]
            if above.size:
                qrs_duration[i] = (above[-1] - above[0]) / fs
            else:
                qrs_duration[i] = 0.0
            pre_window = max(1, pk // 8)
            post_window = max(1, (L - pk) // 8)
            pre_slope[i] = (x[pk] - x[max(0, pk - pre_window)]) / (pre_window / fs)
            post_slope[i] = (x[min(L - 1, pk + post_window)] - x[pk]) / (post_window / fs)

            finite_mask = np.isfinite(x) & np.isfinite(template)
            if np.count_nonzero(finite_mask) < 2:
                corr_template[i] = 0.0
            else:
                t = template[finite_mask] - np.nanmean(template[finite_mask])
                xx = x[finite_mask] - np.nanmean(x[finite_mask])
                denom = np.sqrt(np.sum(t ** 2) * np.sum(xx ** 2))
                corr_template[i] = np.dot(t, xx) / denom if denom > 1e-12 else 0.0
                corr_template[i] = float(np.clip(corr_template[i], -1.0, 1.0))
        except Exception:
            qrs_duration[i] = np.nan
            pre_slope[i] = np.nan
            post_slope[i] = np.nan
            corr_template[i] = np.nan

        # Spectral features
        try:
            f, Pxx = welch(x, fs=fs, nperseg=min(256, L))
            total_power = Pxx.sum()
            if total_power > 0:
                centroid[i] = np.sum(f * Pxx) / total_power
                bandwidth[i] = np.sqrt(np.sum(((f - centroid[i]) ** 2) * Pxx) / total_power)
                dom_freq[i] = f[np.argmax(Pxx)]
                band_0_5[i] = Pxx[(f >= 0) & (f < 5)].sum()
                band_5_15[i] = Pxx[(f >= 5) & (f < 15)].sum()
                band_15_40[i] = Pxx[(f >= 15) & (f < 40)].sum()
            else:
                centroid[i] = np.nan
                bandwidth[i] = np.nan
                dom_freq[i] = np.nan
        except Exception:
            centroid[i] = np.nan
            bandwidth[i] = np.nan
            dom_freq[i] = np.nan

        # Sample entropy
        try:
            sampen_vals[i] = _sample_entropy(x)
        except Exception:
            sampen_vals[i] = np.nan

        # Wavelet energies
        try:
            coeffs = pywt.wavedec(x, wavelet, level=max_level)
            for lvl in range(1, max_level + 1):
                c = coeffs[lvl]
                wave_energies[i, lvl - 1] = np.sum(np.array(c) ** 2)
        except Exception:
            wave_energies[i, :] = 0.0

        _print_progress(i + 1, n_samples, prefix='Waveform beats')

    _finish_progress()

    band_total = band_0_5 + band_5_15 + band_15_40
    band_0_5_norm = np.divide(band_0_5, band_total, out=np.zeros_like(band_0_5), where=band_total > 0)
    band_5_15_norm = np.divide(band_5_15, band_total, out=np.zeros_like(band_5_15), where=band_total > 0)
    band_15_40_norm = np.divide(band_15_40, band_total, out=np.zeros_like(band_15_40), where=band_total > 0)

    corr_template = np.where(np.isfinite(corr_template), corr_template, 0.0)
    diff1 = np.diff(X_segments, axis=1)
    var0 = X_segments.var(axis=1)
    var1 = diff1.var(axis=1)
    activity = var0
    mobility = np.sqrt(var1 / (var0 + 1e-12))
    diff2 = np.diff(diff1, axis=1)
    var2 = diff2.var(axis=1)
    complexity = np.sqrt(var2 / (var1 + 1e-12)) / (mobility + 1e-12)

    if rr_prev is None:
        rr_prev_vals = np.full(n_samples, np.nan, dtype=float)
    else:
        rr_prev_vals = np.asarray(rr_prev, dtype=float).reshape(-1)
        if rr_prev_vals.shape[0] != n_samples:
            raise ValueError(f"rr_prev length {rr_prev_vals.shape[0]} does not match number of samples {n_samples}")

    if rr_next is None:
        rr_next_vals = np.full(n_samples, np.nan, dtype=float)
    else:
        rr_next_vals = np.asarray(rr_next, dtype=float).reshape(-1)
        if rr_next_vals.shape[0] != n_samples:
            raise ValueError(f"rr_next length {rr_next_vals.shape[0]} does not match number of samples {n_samples}")

    features = np.vstack([
        mean, std, ptp, mx, mn, med, p25, p75,
        energy, rms, zero_cross, max_slope, area,
        qrs_amp, r_idx_rel, qrs_duration, pre_slope, post_slope, corr_template,
        sk, kt,
        centroid, bandwidth, dom_freq,
        band_0_5, band_5_15, band_15_40, band_0_5_norm, band_5_15_norm, band_15_40_norm,
        sampen_vals, activity, mobility, complexity,
        wave_energies.T,
        rr_prev_vals, rr_next_vals,
    ]).T.astype(np.float32)

    return features


def get_waveform_feature_names_and_units():
    feature_names = [
        'wf_mean','wf_std','wf_ptp','wf_max','wf_min','wf_median','wf_p25','wf_p75',
        'wf_energy','wf_rms','wf_zero_cross','wf_max_slope','wf_area',
        'qrs_amplitude','r_peak_index_rel','qrs_duration_sec','pre_slope','post_slope','corr_template',
        'wf_skew','wf_kurtosis',
        'spec_centroid','spec_bandwidth','spec_domfreq',
        'band_0_5','band_5_15','band_15_40','band_0_5_norm','band_5_15_norm','band_15_40_norm',
        'sampen','hjorth_activity','hjorth_mobility','hjorth_complexity',
        'wave_energy_lvl1','wave_energy_lvl2','wave_energy_lvl3',
        'rr_prev','rr_next'
    ]
    units = [
        'mV','mV','mV','mV','mV','mV','mV','mV',
        'mV^2*s','mV','count','mV/s','mV*s',
        'mV','fraction','s','mV/s','mV/s','unitless',
        'unitless','unitless',
        'Hz','Hz','Hz',
        'power','power','power','ratio','ratio','ratio',
        'unitless','mV^2','unitless','unitless',
        'mV^2','mV^2','mV^2',
        's','s'
    ]
    explanations = [
        'Mean amplitude of the beat segment',
        'Standard deviation of the beat segment',
        'Peak-to-peak amplitude of the beat segment',
        'Maximum amplitude of the beat segment',
        'Minimum amplitude of the beat segment',
        'Median amplitude of the beat segment',
        '25th percentile amplitude of the beat segment',
        '75th percentile amplitude of the beat segment',
        'Energy of the beat segment',
        'Root mean square amplitude',
        'Number of zero crossings',
        'Maximum absolute slope',
        'Integrated area under the beat segment',
        'Approximate QRS peak amplitude relative to the median',
        'Relative position of the dominant peak within the beat',
        'Approximate QRS duration',
        'Slope before the dominant peak',
        'Slope after the dominant peak',
        'Correlation with a template beat',
        'Skewness of the beat waveform',
        'Kurtosis of the beat waveform',
        'Spectral centroid',
        'Spectral bandwidth',
        'Dominant spectral frequency',
        'Power in the 0-5 Hz band',
        'Power in the 5-15 Hz band',
        'Power in the 15-40 Hz band',
        'Normalized power in the 0-5 Hz band',
        'Normalized power in the 5-15 Hz band',
        'Normalized power in the 15-40 Hz band',
        'Sample entropy of the beat',
        'Hjorth activity',
        'Hjorth mobility',
        'Hjorth complexity',
        'Wavelet energy at level 1',
        'Wavelet energy at level 2',
        'Wavelet energy at level 3',
        'Previous RR interval (time between two successive R peaks)',
        'Next RR interval'
    ]
    return feature_names, units, explanations


def load_raw_workbook(input_excel_path):
    input_excel_path = Path(input_excel_path)
    print(f"Reading raw workbook: {input_excel_path}")
    with pd.ExcelFile(input_excel_path, engine='openpyxl') as excel_file:
        features = pd.read_excel(excel_file, sheet_name='features')
        labels = pd.read_excel(excel_file, sheet_name='labels')
        metadata = pd.read_excel(excel_file, sheet_name='metadata')
    print(f"Loaded {len(features)} rows from features sheet and {len(metadata)} rows from metadata sheet.")
    return features, labels, metadata


def write_raw_workbook_with_rr_flag(features, labels, metadata, output_excel_path=None, rr_problem_flag=None):
    if output_excel_path is None:
        output_excel_path = output_dir / 'mitbih_raw_data_rrflag.xlsx'
    output_excel_path = Path(output_excel_path)

    if rr_problem_flag is None:
        record_ids = metadata['record_id'].to_numpy(dtype=int)
        sample_positions = metadata['sample_position'].to_numpy(dtype=float)
        _, _, rr_problem_flag = compute_rr_intervals(sample_positions, record_ids, fs=360.0)

    metadata_with_flag = metadata.copy(deep=False)
    metadata_with_flag['rr_problem_flag'] = rr_problem_flag

    output_excel_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing updated raw workbook: {output_excel_path}")
    with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
        features.to_excel(writer, sheet_name='features', index=False)
        labels.to_excel(writer, sheet_name='labels', index=False)
        metadata_with_flag.to_excel(writer, sheet_name='metadata', index=False)

    print(f"Saved updated raw workbook to {output_excel_path}")
    return output_excel_path


def compute_and_save_derived_features(features, metadata, output_excel_path=None, rr_problem_flag=None, rr_prev=None, rr_next=None):
    if output_excel_path is None:
        output_excel_path = output_dir / 'derived_waveform_features.xlsx'
    output_excel_path = Path(output_excel_path)

    X = features.to_numpy(dtype=np.float32)
    record_ids = metadata['record_id'].to_numpy(dtype=int, copy=False)
    sample_positions = metadata['sample_position'].to_numpy(dtype=float, copy=False)

    if rr_prev is None or rr_next is None:
        print("Computing RR intervals...")
        rr_prev, rr_next, rr_problem_flag = compute_rr_intervals(sample_positions, record_ids, fs=360.0)
    elif rr_problem_flag is None:
        rr_problem_flag = np.full(len(metadata), 'ok', dtype=object)

    print("Computing derived waveform features...")
    derived_features = compute_waveform_features(X, fs=360.0, rr_prev=rr_prev, rr_next=rr_next)
    feature_names, feature_units, feature_explanations = get_waveform_feature_names_and_units()

    feature_df = pd.DataFrame(derived_features, columns=feature_names)
    metadata_df = metadata.copy(deep=False)
    metadata_df['rr_problem_flag'] = rr_problem_flag
    feature_definitions_df = pd.DataFrame({
        'feature': feature_names,
        'unit': feature_units,
        'explanation': feature_explanations,
    })

    output_excel_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing output workbook: {output_excel_path}")
    with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
        feature_df.to_excel(writer, sheet_name='features', index=False)
        metadata_df.to_excel(writer, sheet_name='metadata', index=False)
        feature_definitions_df.to_excel(writer, sheet_name='feature_definitions', index=False)

    print(f"Saved derived waveform features to {output_excel_path}")
    return output_excel_path, feature_df


if __name__ == '__main__':
    raw_excel = current_dir / 'input' / 'mitbih_raw_data.xlsx'
    output_raw_excel = output_dir / 'mitbih_raw_data_rrflag.xlsx'
    output_derived_excel = output_dir / 'derived_waveform_features.xlsx'
    #raw_excel = current_dir / 'input' / 'mitbih_raw_data_short.xlsx' # 1-record data for testing
    if raw_excel.exists():
        features, labels, metadata = load_raw_workbook(raw_excel)
        record_ids = metadata['record_id'].to_numpy(dtype=int, copy=False)
        sample_positions = metadata['sample_position'].to_numpy(dtype=float, copy=False)
        rr_prev, rr_next, rr_problem_flag = compute_rr_intervals(sample_positions, record_ids, fs=360.0)
        
        compute_and_save_derived_features(
            features,
            metadata,
            output_derived_excel,
            rr_problem_flag=rr_problem_flag,
            rr_prev=rr_prev,
            rr_next=rr_next,
        )
        write_raw_workbook_with_rr_flag(features, labels, metadata, output_raw_excel, rr_problem_flag=rr_problem_flag)
    else:
        print(f"Raw WFDB Excel not found: {raw_excel}")
