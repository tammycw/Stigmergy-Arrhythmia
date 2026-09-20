# Tammy Wang | OSSM | 07/11/26 | tammycc.wang@gmail.com

"""Download MIT-BIH WFDB records and export beat-level ECG data to Excel.

This script reads selected PhysioNet MIT-BIH records, extracts labeled beats,
and saves the resulting feature matrix, metadata, and timeline information to a
workbook for downstream waveform-feature processing.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb


def _print_progress(current, total, prefix='Progress'):
    if total <= 0:
        return
    done = int(20 * current / total)
    bar = '#' * done + '-' * (20 - done)
    print(f"\r{prefix}: [{bar}] {current}/{total}", end='', flush=True)

current_dir = Path(__file__).resolve().parent
# Store the generated workbook in the input directory so it can be reused in
# subsequent preprocessing steps. No raw input data is required here.
output_dir = current_dir / 'input'
output_dir.mkdir(parents=True, exist_ok=True)

DEFAULT_RECORD_NUMBERS = [100, 101, 102, 103, 105, 106, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 220, 221, 222, 223, 228, 230, 231, 232, 233, 234]


def download_wfdb_to_excel(record_numbers=None, samples_per_beat=180, max_samples=None, output_path=None):
    if record_numbers is None:
        record_numbers = DEFAULT_RECORD_NUMBERS

    X = []
    y = []
    record_ids = []
    beat_classes = []
    sample_positions = []
    timelines = []

    total_records = len(record_numbers)
    print(f"Starting WFDB download for {total_records} record(s)...")

    for idx, rec in enumerate(record_numbers, start=1):
        print(f"\nProcessing record {rec} ({idx}/{total_records})...")
        try:
            record = wfdb.rdrecord(str(rec), pn_dir='mitdb')
            annotation = wfdb.rdann(str(rec), 'atr', pn_dir='mitdb')
            signal = record.p_signal[:, 0]

            accepted_beats = 0
            total_beats = len(annotation.symbol)
            for i, symbol in enumerate(annotation.symbol):
                if symbol not in ['N', 'V', 'S', 'F']:
                    continue

                center = int(annotation.sample[i])
                half = samples_per_beat // 2
                if center - half < 0 or center + half >= len(signal):
                    continue

                segment = signal[center - half:center + half]
                X.append(segment)
                y.append(0 if symbol == 'N' else 1)
                record_ids.append(rec)
                beat_classes.append(symbol)
                sample_positions.append(center)
                timelines.append(accepted_beats)
                accepted_beats += 1

                if max_samples is not None and len(X) >= max_samples:
                    break
            print(f"Collected {accepted_beats} qualifying beats from record {rec}.")
            if max_samples is not None and len(X) >= max_samples:
                break
        except Exception as exc:
            print(f"Error loading record {rec}: {exc}")

    print()
    _print_progress(len(X), max_samples or len(X), prefix='Downloaded beats')
    print()

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    record_ids = np.asarray(record_ids, dtype=int)
    beat_classes = np.asarray(beat_classes, dtype=object)
    sample_positions = np.asarray(sample_positions, dtype=float)
    timelines = np.asarray(timelines, dtype=int)

    feature_df = pd.DataFrame(X, dtype=float)
    label_df = pd.DataFrame(y.reshape(-1, 1), columns=['label'])
    metadata_df = pd.DataFrame({
        'record_id': record_ids,
        'beat_class': beat_classes,
        'label': y.reshape(-1),
        'sample_position': sample_positions,
        'timeline': timelines,
    })

    if output_path is None:
        output_path = output_dir / 'mitbih_raw_data.xlsx'
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path) as writer:
        feature_df.to_excel(writer, sheet_name='features', index=False)
        label_df.to_excel(writer, sheet_name='labels', index=False)
        metadata_df.to_excel(writer, sheet_name='metadata', index=False)

    print(f"Saved raw WFDB data to {output_path}")
    return output_path, feature_df, label_df, metadata_df


if __name__ == '__main__':
    download_wfdb_to_excel()
