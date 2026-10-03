# Tammy Wang | OSSM | 07/11/2026 | tammycc.wang@gmail.com

"""Compute leakage-safe derived waveform features from ECG beat segments.

Updated: 09/10/2026
This module is a leakage-safe successor to the earlier template-based feature
builder. It keeps per-beat waveform, spectral, complexity, and RR features, but
removes the workbook-wide leakage patterns that can otherwise contaminate
patient-disjoint evaluation.

The feature extraction is intentionally designed for inter-patient validation:

- omits ``corr_template`` because the original template was computed across all
  records, including held-out patients; template similarity should instead be
  computed from training-fold templates during model validation;
- measures QRS width as the contiguous threshold-crossing region around the
  dominant peak, rather than spanning every above-threshold sample in a beat;
- adds RR features relative to each record's local rhythm context, computed
  without using beat labels or pooled patient information.

The script reads beat-level ECG segments from an Excel workbook and enriches
those rows with handcrafted features suitable for downstream classification. The
feature families include:

- waveform statistics: mean, spread, range, percentiles, energy, RMS, zero
  crossings, slope, and area;
- QRS characterization: peak position, amplitude, duration, and pre-/post-peak
  slopes near the dominant beat peak;
- spectral descriptors: centroid, bandwidth, dominant frequency, and power in
  fixed frequency bands;
- nonlinear and complexity measures: sample entropy and Hjorth-based activity,
  mobility, and complexity;
- wavelet features: multiscale energy content from discrete wavelet
  decomposition;
- RR intervals: previous and next beat-to-beat intervals derived from sample
  positions within each recording.

The resulting output is written to an Excel workbook with separate sheets for
feature values, metadata, and feature definitions.

Input:  input/mitbih_raw_data.xlsx
Output: input/derived_waveform_features.xlsx
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pywt
from scipy.signal import welch
from scipy.stats import kurtosis, skew


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


CURRENT_DIR = Path(__file__).resolve().parent
INPUT_PATH = CURRENT_DIR / "input" / "mitbih_raw_data.xlsx"
OUTPUT_PATH = CURRENT_DIR / "input" / "derived_waveform_features.xlsx"
SAMPLE_RATE = 360.0
RR_CONTEXT_INTERVALS = 5
QRS_WIDTH_FRACTIONS = (0.35, 0.50)


def contiguous_qrs_width(envelope, peak_index, fraction, sample_rate):
    """Return the threshold width of the contiguous region around the peak."""
    peak_value = float(envelope[peak_index])
    if not np.isfinite(peak_value) or peak_value <= 0:
        return 0.0

    threshold = peak_value * fraction
    left = peak_index
    right = peak_index
    while left > 0 and envelope[left - 1] >= threshold:
        left -= 1
    while right + 1 < len(envelope) and envelope[right + 1] >= threshold:
        right += 1
    return (right - left + 1) / sample_rate


def compute_local_rr_features(sample_positions, record_ids, sample_rate=SAMPLE_RATE):
    """Compute RR intervals and local relative-rhythm features per record."""
    sample_positions = np.asarray(sample_positions, dtype=float)
    record_ids = np.asarray(record_ids)
    rr_previous = np.full(len(record_ids), np.nan, dtype=float)
    rr_next = np.full(len(record_ids), np.nan, dtype=float)
    rr_previous_ratio = np.full(len(record_ids), np.nan, dtype=float)
    rr_next_ratio = np.full(len(record_ids), np.nan, dtype=float)
    rr_local_cv = np.full(len(record_ids), np.nan, dtype=float)

    for record_id in np.unique(record_ids):
        record_indices = np.flatnonzero(record_ids == record_id)
        valid = np.isfinite(sample_positions[record_indices])
        record_indices = record_indices[valid]
        if len(record_indices) < 2:
            continue

        order = np.argsort(sample_positions[record_indices], kind="stable")
        sorted_indices = record_indices[order]
        rr_intervals = np.diff(sample_positions[sorted_indices]) / sample_rate
        rr_intervals[~np.isfinite(rr_intervals) | (rr_intervals <= 0)] = np.nan
        n_intervals = len(rr_intervals)

        for beat_position, row_index in enumerate(sorted_indices):
            previous_interval = beat_position - 1
            next_interval = beat_position
            if previous_interval >= 0:
                rr_previous[row_index] = rr_intervals[previous_interval]
            if next_interval < n_intervals:
                rr_next[row_index] = rr_intervals[next_interval]

            start = max(0, beat_position - RR_CONTEXT_INTERVALS // 2)
            stop = min(n_intervals, beat_position + RR_CONTEXT_INTERVALS // 2 + 1)
            context = rr_intervals[start:stop]
            context = context[np.isfinite(context)]
            if context.size == 0:
                continue
            local_median = float(np.median(context))
            if local_median > 0:
                if np.isfinite(rr_previous[row_index]):
                    rr_previous_ratio[row_index] = rr_previous[row_index] / local_median
                if np.isfinite(rr_next[row_index]):
                    rr_next_ratio[row_index] = rr_next[row_index] / local_median
            local_mean = float(np.mean(context))
            if context.size > 1 and local_mean > 0:
                rr_local_cv[row_index] = float(np.std(context) / local_mean)

    return {
        "rr_prev": rr_previous,
        "rr_next": rr_next,
        "rr_prev_local_ratio": rr_previous_ratio,
        "rr_next_local_ratio": rr_next_ratio,
        "rr_local_cv": rr_local_cv,
    }


def compute_features(waveforms, rr_features, sample_rate=SAMPLE_RATE):
    """Extract beat-level features without a dataset-fitted template."""
    waveforms = np.asarray(waveforms, dtype=float)
    if waveforms.ndim != 2:
        raise ValueError(f"Expected a 2D waveform matrix, got shape {waveforms.shape}.")
    n_beats, n_samples = waveforms.shape

    names = [
        "wf_mean", "wf_std", "wf_ptp", "wf_max", "wf_min", "wf_median",
        "wf_p25", "wf_p75", "wf_energy", "wf_rms", "wf_zero_cross",
        "wf_max_slope", "wf_area", "qrs_amplitude", "r_peak_index_rel",
        "qrs_width_35_sec", "qrs_width_50_sec", "pre_slope", "post_slope",
        "wf_skew", "wf_kurtosis", "spec_centroid", "spec_bandwidth",
        "spec_domfreq", "band_0_5", "band_5_15", "band_15_40",
        "band_0_5_norm", "band_5_15_norm", "band_15_40_norm", "sampen",
        "hjorth_activity", "hjorth_mobility", "hjorth_complexity",
        "wave_energy_lvl1", "wave_energy_lvl2", "wave_energy_lvl3",
        "rr_prev", "rr_next", "rr_prev_local_ratio", "rr_next_local_ratio",
        "rr_local_cv",
    ]
    values = np.full((n_beats, len(names)), np.nan, dtype=np.float64)

    for beat_index, waveform in enumerate(waveforms):
        x = np.asarray(waveform, dtype=float)
        finite = np.isfinite(x)
        if not finite.all():
            if not finite.any():
                raise ValueError(f"Beat {beat_index} contains no finite waveform samples.")
            x = np.interp(np.arange(n_samples), np.flatnonzero(finite), x[finite])

        median = float(np.median(x))
        centered = x - median
        envelope = np.abs(centered)
        peak_index = int(np.argmax(envelope))
        peak_amplitude = float(envelope[peak_index])

        first_difference = np.diff(x)
        second_difference = np.diff(first_difference)
        variance_0 = float(np.var(x))
        variance_1 = float(np.var(first_difference))
        variance_2 = float(np.var(second_difference))

        frequency, power = welch(x, fs=sample_rate, nperseg=min(256, n_samples))
        total_power = float(power.sum())
        bands = [
            float(power[(frequency >= low) & (frequency < high)].sum())
            for low, high in ((0, 5), (5, 15), (15, 40))
        ]
        band_total = sum(bands)
        if total_power > 0:
            centroid = float(np.sum(frequency * power) / total_power)
            bandwidth = float(np.sqrt(np.sum(((frequency - centroid) ** 2) * power) / total_power))
            dominant_frequency = float(frequency[np.argmax(power)])
        else:
            centroid = bandwidth = dominant_frequency = np.nan

        try:
            wavelet_coefficients = pywt.wavedec(x, "db4", level=3)
            wavelet_energies = [float(np.sum(coefficients ** 2)) for coefficients in wavelet_coefficients[1:4]]
        except (ValueError, FloatingPointError):
            wavelet_energies = [np.nan, np.nan, np.nan]

        row = {
            "wf_mean": float(np.mean(x)),
            "wf_std": float(np.std(x)),
            "wf_ptp": float(np.ptp(x)),
            "wf_max": float(np.max(x)),
            "wf_min": float(np.min(x)),
            "wf_median": median,
            "wf_p25": float(np.percentile(x, 25)),
            "wf_p75": float(np.percentile(x, 75)),
            "wf_energy": float(np.sum(x ** 2)),
            "wf_rms": float(np.sqrt(np.mean(x ** 2))),
            "wf_zero_cross": float(np.sum(x[:-1] * x[1:] < 0)),
            "wf_max_slope": float(np.max(np.abs(first_difference)) * sample_rate),
            "wf_area": float(np.sum(x) / sample_rate),
            "qrs_amplitude": peak_amplitude,
            "r_peak_index_rel": peak_index / max(n_samples - 1, 1),
            "qrs_width_35_sec": contiguous_qrs_width(envelope, peak_index, QRS_WIDTH_FRACTIONS[0], sample_rate),
            "qrs_width_50_sec": contiguous_qrs_width(envelope, peak_index, QRS_WIDTH_FRACTIONS[1], sample_rate),
            "pre_slope": np.nan,
            "post_slope": np.nan,
            "wf_skew": float(skew(x, bias=True)),
            "wf_kurtosis": float(kurtosis(x, bias=True)),
            "spec_centroid": centroid,
            "spec_bandwidth": bandwidth,
            "spec_domfreq": dominant_frequency,
            "band_0_5": bands[0],
            "band_5_15": bands[1],
            "band_15_40": bands[2],
            "band_0_5_norm": bands[0] / band_total if band_total > 0 else 0.0,
            "band_5_15_norm": bands[1] / band_total if band_total > 0 else 0.0,
            "band_15_40_norm": bands[2] / band_total if band_total > 0 else 0.0,
            "sampen": _sample_entropy(x),
            "hjorth_activity": variance_0,
            "hjorth_mobility": float(np.sqrt(variance_1 / (variance_0 + 1e-12))),
            "hjorth_complexity": float(
                np.sqrt(variance_2 / (variance_1 + 1e-12))
                / (np.sqrt(variance_1 / (variance_0 + 1e-12)) + 1e-12)
            ),
            "wave_energy_lvl1": wavelet_energies[0],
            "wave_energy_lvl2": wavelet_energies[1],
            "wave_energy_lvl3": wavelet_energies[2],
        }

        slope_window = max(1, int(round(0.02 * sample_rate)))
        if peak_index >= slope_window:
            row["pre_slope"] = float(
                (centered[peak_index] - centered[peak_index - slope_window])
                * sample_rate / slope_window
            )
        else:
            row["pre_slope"] = 0.0
        if peak_index + slope_window < n_samples:
            row["post_slope"] = float(
                (centered[peak_index + slope_window] - centered[peak_index])
                * sample_rate / slope_window
            )
        else:
            row["post_slope"] = 0.0

        for name, feature_values in rr_features.items():
            row[name] = float(feature_values[beat_index])
        values[beat_index] = [row[name] for name in names]
        _print_progress(beat_index + 1, n_beats, prefix="Leakage-safe feature beats")

    _finish_progress()
    return pd.DataFrame(values, columns=names)


def feature_definitions():
    base_names, base_units, base_explanations = get_waveform_feature_names_and_units()
    units = dict(zip(base_names, base_units))
    explanations = dict(zip(base_names, base_explanations))
    units.pop("qrs_duration_sec", None)
    units.pop("corr_template", None)
    explanations.pop("qrs_duration_sec", None)
    explanations.pop("corr_template", None)

    units.update({
        "qrs_width_35_sec": "s",
        "qrs_width_50_sec": "s",
        "rr_prev_local_ratio": "ratio",
        "rr_next_local_ratio": "ratio",
        "rr_local_cv": "ratio",
    })
    explanations = {
        **explanations,
        "qrs_amplitude": "Absolute baseline-corrected amplitude of the dominant beat peak",
        "r_peak_index_rel": "Relative position of the dominant peak within the beat segment",
        "qrs_width_35_sec": "Contiguous width around the dominant peak above 35% of its baseline-corrected amplitude",
        "qrs_width_50_sec": "Contiguous width around the dominant peak above 50% of its baseline-corrected amplitude",
        "pre_slope": "Slope over the 20 ms preceding the dominant peak",
        "post_slope": "Slope over the 20 ms following the dominant peak",
        "rr_prev_local_ratio": "Previous RR interval divided by the local median of nearby RR intervals",
        "rr_next_local_ratio": "Next RR interval divided by the local median of nearby RR intervals; uses one-beat lookahead",
        "rr_local_cv": "Coefficient of variation of nearby RR intervals within the same record",
    }
    units["pre_slope"] = "mV/s"
    units["post_slope"] = "mV/s"
    return explanations, units


def write_raw_workbook_with_rr_flag(features, labels, metadata, output_excel_path=None, rr_problem_flag=None):
    if output_excel_path is None:
        output_excel_path = CURRENT_DIR / "input" / "mitbih_raw_data_rrflag.xlsx"
    output_excel_path = Path(output_excel_path)

    if rr_problem_flag is None:
        record_ids = metadata["record_id"].to_numpy(dtype=int)
        sample_positions = metadata["sample_position"].to_numpy(dtype=float)
        _, _, rr_problem_flag = compute_rr_intervals(sample_positions, record_ids, fs=SAMPLE_RATE)

    metadata_with_flag = metadata.copy(deep=False)
    metadata_with_flag["rr_problem_flag"] = rr_problem_flag

    output_excel_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing updated raw workbook: {output_excel_path}")
    with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
        features.to_excel(writer, sheet_name="features", index=False)
        labels.to_excel(writer, sheet_name="labels", index=False)
        metadata_with_flag.to_excel(writer, sheet_name="metadata", index=False)

    print(f"Saved updated raw workbook to {output_excel_path}")
    return output_excel_path


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Raw ECG workbook not found: {INPUT_PATH}")

    waveforms, labels, metadata = load_raw_workbook(INPUT_PATH)
    if len(waveforms) != len(metadata) or len(labels) != len(metadata):
        raise ValueError("Waveform, label, and metadata row counts do not match.")

    record_ids = metadata["record_id"].to_numpy(dtype=int)
    sample_positions = metadata["sample_position"].to_numpy(dtype=float)
    rr_previous, rr_next, rr_problem_flag = compute_rr_intervals(
        sample_positions, record_ids, fs=SAMPLE_RATE
    )
    rr_features = compute_local_rr_features(sample_positions, record_ids, SAMPLE_RATE)
    rr_features["rr_prev"] = rr_previous
    rr_features["rr_next"] = rr_next
    derived_features = compute_features(
        waveforms.to_numpy(dtype=np.float32), rr_features, sample_rate=SAMPLE_RATE
    )

    metadata_output = metadata.copy()
    metadata_output["rr_problem_flag"] = rr_problem_flag
    definitions, units = feature_definitions()
    missing_definitions = set(derived_features.columns) - set(units)
    if missing_definitions:
        raise ValueError(f"Missing feature units: {sorted(missing_definitions)}")
    missing_explanations = set(derived_features.columns) - set(definitions)
    if missing_explanations:
        raise ValueError(f"Missing feature explanations: {sorted(missing_explanations)}")
    feature_definitions_frame = pd.DataFrame({
        "feature": derived_features.columns,
        "unit": [units[name] for name in derived_features.columns],
        "explanation": [definitions[name] for name in derived_features.columns],
    })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        derived_features.to_excel(writer, sheet_name="features", index=False)
        metadata_output.to_excel(writer, sheet_name="metadata", index=False)
        labels.to_excel(writer, sheet_name="labels", index=False)
        feature_definitions_frame.to_excel(writer, sheet_name="feature_definitions", index=False)

    raw_output_path = CURRENT_DIR / "input" / "mitbih_raw_data_rrflag.xlsx"
    write_raw_workbook_with_rr_flag(
        waveforms,
        labels,
        metadata,
        output_excel_path=raw_output_path,
        rr_problem_flag=rr_problem_flag,
    )

    print(f"Generated {len(derived_features)} rows and {len(derived_features.columns)} leakage-safe features.")
    print("Class counts:")
    print(metadata_output["beat_class"].value_counts().sort_index().to_string())
    print(f"Template similarity omitted; compute training-fold templates inside model validation.")
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Saved raw RR-flag workbook: {raw_output_path}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()