from pathlib import Path
import traceback

import neurokit2 as nk
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_ROOT = Path("./ecg_signals_lmu")
OUTPUT_ROOT = Path("./ecg_intervals_lmu")

NPY_FILENAME = "ecg_12lead.npy"
OUTPUT_EXCEL_NAME = "ecg.xlsx"

# The NPY file contains only the signal array, so lead order and sample
# rate must match the values used by the extraction script.
LEAD_NAMES = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]

SAMPLE_RATE_HZ = 500.0

# Leads are evaluated in this order.
# Lead II is usually useful for P-wave detection.
LEAD_PREFERENCE = [
    "II",
    "I",
    "aVF",
    "V5",
    "V6",
    "V4",
    "III",
    "aVL",
    "V2",
    "V3",
    "V1",
    "aVR",
]

DELINEATION_METHOD = "dwt"

MIN_SIGNAL_DURATION_SECONDS = 2.0
MIN_R_PEAKS = 2

# At least this many valid beat measurements are required from one lead.
MIN_VALID_BEATS_PER_LEAD = 2

# Physiological/error-rejection ranges.
MIN_QRS_DURATION_MS = 40
MAX_QRS_DURATION_MS = 300

MIN_PQ_INTERVAL_MS = 60
MAX_PQ_INTERVAL_MS = 500


# ============================================================
# BASIC UTILITIES
# ============================================================

def safe_float(value):
    """Convert a value to a finite float or return None."""
    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except (TypeError, ValueError):
        pass

    return None


def sanitize_indices(values, signal_length):
    """
    Convert NeuroKit boundary values into valid integer indices.
    """
    if values is None:
        return np.array([], dtype=int)

    cleaned = []

    for value in values:
        value = safe_float(value)

        if value is None:
            continue

        index = int(round(value))

        if 0 <= index < signal_length:
            cleaned.append(index)

    return np.asarray(
        cleaned,
        dtype=int,
    )


def load_ecg_npy(npy_path):
    """
    Load one standardized 12-lead ECG NPY file.

    Expected shape:
        [12, time]
    """
    signals = np.load(
        npy_path,
        allow_pickle=False,
    )

    signals = np.asarray(
        signals,
        dtype=np.float64,
    )

    if signals.ndim != 2:
        raise ValueError(
            f"Expected signals with shape [12, time], "
            f"but received {signals.shape}"
        )

    if signals.shape[0] != len(LEAD_NAMES):
        raise ValueError(
            f"Expected {len(LEAD_NAMES)} leads, "
            f"but found {signals.shape[0]} in {npy_path}"
        )

    # A lead is considered available when it contains at least one
    # finite value. Fully missing leads were saved as NaN rows.
    available_leads = np.asarray(
        [
            np.isfinite(signal).any()
            for signal in signals
        ],
        dtype=bool,
    )

    return (
        signals,
        LEAD_NAMES.copy(),
        available_leads,
        SAMPLE_RATE_HZ,
    )


def prepare_signal(signal):
    """
    Interpolate missing regions and reject unusable traces.
    """
    signal = np.asarray(
        signal,
        dtype=np.float64,
    )

    valid = np.isfinite(signal)

    # Entirely missing lead.
    if valid.sum() == 0:
        return None

    # Reject traces where more than 10% of values are missing.
    if valid.mean() < 0.90:
        return None

    if not valid.all():
        positions = np.arange(
            len(signal)
        )

        signal = np.interp(
            positions,
            positions[valid],
            signal[valid],
        )

    # Remove constant baseline offset.
    signal = signal - np.median(signal)

    # Reject effectively flat signals.
    if np.std(signal) < 1e-5:
        return None

    return signal


# ============================================================
# BOUNDARY PAIRING
# ============================================================

def find_nearest_before(
    indices,
    reference,
    maximum_distance=None,
):
    """Return the nearest index occurring before the reference."""
    candidates = indices[
        indices < reference
    ]

    if len(candidates) == 0:
        return None

    selected = int(
        candidates[-1]
    )

    if (
        maximum_distance is not None
        and reference - selected > maximum_distance
    ):
        return None

    return selected


def find_nearest_after(
    indices,
    reference,
    maximum_distance=None,
):
    """Return the nearest index occurring after the reference."""
    candidates = indices[
        indices > reference
    ]

    if len(candidates) == 0:
        return None

    selected = int(
        candidates[0]
    )

    if (
        maximum_distance is not None
        and selected - reference > maximum_distance
    ):
        return None

    return selected


def calculate_intervals(
    r_peaks,
    qrs_onsets,
    qrs_offsets,
    p_onsets,
    sample_rate,
):
    """
    Calculate valid beat-level intervals.

    QRS duration:
        QRS offset - QRS onset

    PQ/PR interval:
        QRS onset - P-wave onset
    """
    qrs_values = []
    pq_values = []

    maximum_qrs_side_samples = int(
        round(
            0.25 * sample_rate
        )
    )

    maximum_p_to_qrs_samples = int(
        round(
            0.60 * sample_rate
        )
    )

    for r_peak in r_peaks:
        qrs_onset = find_nearest_before(
            qrs_onsets,
            r_peak,
            maximum_distance=maximum_qrs_side_samples,
        )

        qrs_offset = find_nearest_after(
            qrs_offsets,
            r_peak,
            maximum_distance=maximum_qrs_side_samples,
        )

        if qrs_onset is None or qrs_offset is None:
            continue

        if qrs_offset <= qrs_onset:
            continue

        qrs_duration_ms = (
            (qrs_offset - qrs_onset)
            / sample_rate
            * 1000.0
        )

        if (
            MIN_QRS_DURATION_MS
            <= qrs_duration_ms
            <= MAX_QRS_DURATION_MS
        ):
            qrs_values.append(
                float(qrs_duration_ms)
            )

        p_onset = find_nearest_before(
            p_onsets,
            qrs_onset,
            maximum_distance=maximum_p_to_qrs_samples,
        )

        if p_onset is None:
            continue

        pq_interval_ms = (
            (qrs_onset - p_onset)
            / sample_rate
            * 1000.0
        )

        if (
            MIN_PQ_INTERVAL_MS
            <= pq_interval_ms
            <= MAX_PQ_INTERVAL_MS
        ):
            pq_values.append(
                float(pq_interval_ms)
            )

    return (
        np.asarray(
            qrs_values,
            dtype=np.float64,
        ),
        np.asarray(
            pq_values,
            dtype=np.float64,
        ),
    )


# ============================================================
# LEAD PROCESSING
# ============================================================

def measure_lead_intervals(
    signal,
    sample_rate,
):
    """
    Extract one median QRS and one median PQ value from one lead.
    """
    signal = prepare_signal(
        signal
    )

    if signal is None:
        return None, None

    minimum_samples = int(
        round(
            MIN_SIGNAL_DURATION_SECONDS
            * sample_rate
        )
    )

    if len(signal) < minimum_samples:
        return None, None

    cleaned = nk.ecg_clean(
        signal,
        sampling_rate=sample_rate,
        method="neurokit",
    )

    _, peak_info = nk.ecg_peaks(
        cleaned,
        sampling_rate=sample_rate,
        method="neurokit",
        correct_artifacts=True,
    )

    r_peaks = sanitize_indices(
        peak_info.get(
            "ECG_R_Peaks"
        ),
        len(cleaned),
    )

    if len(r_peaks) < MIN_R_PEAKS:
        return None, None

    _, waves = nk.ecg_delineate(
        cleaned,
        rpeaks=r_peaks,
        sampling_rate=sample_rate,
        method=DELINEATION_METHOD,
        show=False,
        check=True,
    )

    qrs_onsets = sanitize_indices(
        waves.get(
            "ECG_R_Onsets"
        ),
        len(cleaned),
    )

    qrs_offsets = sanitize_indices(
        waves.get(
            "ECG_R_Offsets"
        ),
        len(cleaned),
    )

    p_onsets = sanitize_indices(
        waves.get(
            "ECG_P_Onsets"
        ),
        len(cleaned),
    )

    qrs_values, pq_values = calculate_intervals(
        r_peaks=r_peaks,
        qrs_onsets=qrs_onsets,
        qrs_offsets=qrs_offsets,
        p_onsets=p_onsets,
        sample_rate=sample_rate,
    )

    qrs_median = None
    pq_median = None

    if len(qrs_values) >= MIN_VALID_BEATS_PER_LEAD:
        qrs_median = float(
            np.median(
                qrs_values
            )
        )

    if len(pq_values) >= MIN_VALID_BEATS_PER_LEAD:
        pq_median = float(
            np.median(
                pq_values
            )
        )

    return (
        qrs_median,
        pq_median,
    )


# ============================================================
# PATIENT PROCESSING
# ============================================================

def extract_patient_id(npy_path):
    """
    Use the NPY parent-folder name as the patient ID.

    Example:
        ecg_signals_lmu/12345/ecg_12lead.npy
        -> ID = 12345
    """
    return npy_path.parent.name


def process_patient(npy_path):
    """
    Produce one integer QRSADM and one integer PQADM value.
    """
    (
        signals,
        lead_names,
        available_leads,
        sample_rate,
    ) = load_ecg_npy(
        npy_path
    )

    lead_to_index = {
        lead_name: index
        for index, lead_name in enumerate(
            lead_names
        )
    }

    qrs_values_across_leads = []
    pq_values_across_leads = []

    for lead_name in LEAD_PREFERENCE:
        if lead_name not in lead_to_index:
            continue

        lead_index = lead_to_index[
            lead_name
        ]

        if not available_leads[
            lead_index
        ]:
            continue

        try:
            (
                qrs_value,
                pq_value,
            ) = measure_lead_intervals(
                signal=signals[
                    lead_index
                ],
                sample_rate=sample_rate,
            )

        except Exception:
            # One failed lead should not stop processing the patient.
            continue

        if qrs_value is not None:
            qrs_values_across_leads.append(
                qrs_value
            )

        if pq_value is not None:
            pq_values_across_leads.append(
                pq_value
            )

    # Produce one patient-level value by taking the median across leads.
    if qrs_values_across_leads:
        qrsadm = int(
            round(
                np.median(
                    qrs_values_across_leads
                )
            )
        )
    else:
        qrsadm = pd.NA

    if pq_values_across_leads:
        pqadm = int(
            round(
                np.median(
                    pq_values_across_leads
                )
            )
        )
    else:
        pqadm = pd.NA

    return {
        "ID": extract_patient_id(
            npy_path
        ),
        "QRSADM": qrsadm,
        "PQADM": pqadm,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: "
            f"{INPUT_ROOT}"
        )

    npy_files = sorted(
        INPUT_ROOT.rglob(
            NPY_FILENAME
        )
    )

    if not npy_files:
        raise RuntimeError(
            f"No files named '{NPY_FILENAME}' were found under "
            f"{INPUT_ROOT}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("QRSADM AND PQADM EXTRACTION")
    print("=" * 80)
    print(f"Input root:      {INPUT_ROOT}")
    print(f"Output root:     {OUTPUT_ROOT}")
    print(f"ECGs discovered: {len(npy_files)}")
    print(f"Sample rate:     {SAMPLE_RATE_HZ} Hz")

    results = []
    failures = []

    for index, npy_path in enumerate(
        npy_files,
        start=1,
    ):
        patient_id = extract_patient_id(
            npy_path
        )

        print(
            f"[{index}/{len(npy_files)}] "
            f"ID={patient_id}"
        )

        try:
            result = process_patient(
                npy_path
            )

            results.append(
                result
            )

            qrs_text = (
                result["QRSADM"]
                if pd.notna(
                    result["QRSADM"]
                )
                else "unavailable"
            )

            pq_text = (
                result["PQADM"]
                if pd.notna(
                    result["PQADM"]
                )
                else "unavailable"
            )

            print(
                f"  QRSADM={qrs_text}, "
                f"PQADM={pq_text}"
            )

        except Exception as error:
            failures.append(
                {
                    "ID": patient_id,
                    "SOURCE_NPY": str(
                        npy_path
                    ),
                    "ERROR": str(
                        error
                    ),
                    "TRACEBACK": traceback.format_exc(),
                }
            )

            print(
                f"  ERROR: {error}"
            )

    results_df = pd.DataFrame(
        results,
        columns=[
            "ID",
            "QRSADM",
            "PQADM",
        ],
    )

    # Nullable integer type keeps valid values as integers while allowing
    # empty Excel cells for unmeasurable intervals.
    if not results_df.empty:
        results_df["QRSADM"] = (
            pd.to_numeric(
                results_df["QRSADM"],
                errors="coerce",
            )
            .round()
            .astype("Int64")
        )

        results_df["PQADM"] = (
            pd.to_numeric(
                results_df["PQADM"],
                errors="coerce",
            )
            .round()
            .astype("Int64")
        )

    output_excel_path = (
        OUTPUT_ROOT
        / OUTPUT_EXCEL_NAME
    )

    results_df.to_excel(
        output_excel_path,
        index=False,
    )

    if failures:
        pd.DataFrame(
            failures
        ).to_csv(
            OUTPUT_ROOT
            / "failed_interval_extractions.csv",
            index=False,
        )

    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(
        f"NPY files discovered:     "
        f"{len(npy_files)}"
    )
    print(
        f"Successfully processed:   "
        f"{len(results)}"
    )
    print(
        f"Failed:                   "
        f"{len(failures)}"
    )

    if not results_df.empty:
        print(
            f"QRSADM extracted:         "
            f"{int(results_df['QRSADM'].notna().sum())}"
        )

        print(
            f"PQADM extracted:          "
            f"{int(results_df['PQADM'].notna().sum())}"
        )

        print(
            f"Both values extracted:    "
            f"{int(
                (
                    results_df['QRSADM'].notna()
                    & results_df['PQADM'].notna()
                ).sum()
            )}"
        )

    print(
        f"Excel saved to:           "
        f"{output_excel_path}"
    )


if __name__ == "__main__":
    main()