#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final TAVI dataset construction / harmonization.

Goals
-----
1. Build the exact same patient cohort for tabular, CT-only and combined models
   by keeping only patients with an available image.
2. Harmonize TUM and LMU tabular definitions before splitting.
3. Remove known high-risk missingness/leakage features from BOTH sites.
4. Keep identical columns and column order for TUM, LMU and merged datasets.
5. Recalculate calcium totals from components consistently.
6. Convert LMU raw perimeter/area values to the derived-diameter definitions
   already used by TUM.
7. Harmonize LVEF top-coding and ECC_INDEX precision.
8. Add QRS/PQ intervals by patient ID.
9. Create one fixed independent test set + five development folds per site.
10. Guarantee merged test/foldX == corresponding TUM + LMU split exactly.
11. Build balanced data-size subsets while keeping the same fixed test set.
12. Save dataset-audit tables so domain/missingness artifacts remain visible.
13. Enforce exact No Event/Pacemaker balance in every fixed test set.
14. Plot post-drop top-10 missingness for TUM, LMU and merged cohorts.
15. Run No Event vs Pacemaker MWU + BH-FDR significance analysis for all three cohorts.
16. Export modelling datasets/splits with exactly AFPRE, BMI, brand2, model4,
    PREDILATION, under a separate output tree.

IMPORTANT
---------
- The script does NOT impute model inputs. Imputation/native-NaN handling belongs
  inside each model pipeline and must be fit using training data only.
- The script does NOT add missingness-mask columns. Known label-dependent
  missingness variables are removed from both sites before model training.
- Every harmonization choice is explicit in the CONFIGURATION section.
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


# =============================================================================
# CONFIGURATION
# =============================================================================

TUM_EXCEL_PATH = Path("./tum.xlsx")
LMU_EXCEL_PATH = Path("./lmu.xlsx")

# Same cohort is used for tabular, CT-only and combined experiments.
IMAGE_ROOT = Path("/home/ubuntu/final_dataset")

TUM_ECG_INTERVAL_PATH = Path(
    "/home/ubuntu/TAVI_new/dataset/dataset_new_binary.xlsx"
)
LMU_ECG_INTERVAL_PATH = Path(
    "/home/ubuntu/TAVI_final/dataset/ecg/ecg_intervals_lmu/ecg.xlsx"
)

# Separate output tree for the 5-feature version so existing outputs are untouched.
OUTPUT_ROOT = Path("./five_feature_dataset")
TUM_CLEANED_OUTPUT_PATH = OUTPUT_ROOT / "tum_cleaned_5_features.xlsx"
LMU_CLEANED_OUTPUT_PATH = OUTPUT_ROOT / "lmu_cleaned_5_features.xlsx"
ENTIRE_OUTPUT_PATH = OUTPUT_ROOT / "entire_5_features.xlsx"
SPLIT_OUTPUT_ROOT = OUTPUT_ROOT / "dataset_splits_5_features"
AUDIT_OUTPUT_ROOT = OUTPUT_ROOT / "dataset_audit"

# IMPORTANT:
# The full harmonized dataframe is still used internally to reproduce the exact
# same cohort, fixed test set, folds, balancing and data-size subsets.
# Only files exported for modelling are reduced to these five columns.
EXPORT_COLUMNS = [
    "ID",
    "LABEL",
    "AFPRE",
    "BMI",
    "brand2",
    "model4",
    "PREDILATION",
]

ID_COLUMN: Optional[str] = "ID"
LABEL_COLUMN = "LABEL"
SEX_COLUMN = "SEX"
QRS_COLUMN = "QRSADM"
PQ_COLUMN = "PQADM"

TEST_FRACTION = 0.20
N_FOLDS = 5
RANDOM_SEED = 42

DATA_SIZE_PERCENTAGES = [2, 5, 10, 20, 50]
DATA_SIZE_OUTPUT_ROOT = SPLIT_OUTPUT_ROOT / "data_size"

# -----------------------------------------------------------------------------
# FINAL HARMONIZATION CHOICES
# -----------------------------------------------------------------------------

# These variables are removed from BOTH sites, preserving an identical schema.
# Rationale:
# - LMU NTPROBNPPRE is essentially unavailable.
# - LMU DM/CABGPRE missingness is strongly label-dependent and can act as a
#   shortcut/leakage signal even when the numerical values are not informative.
DROP_FEATURE_COLUMNS = [
    "DM",
    "CABGPRE",
    "NTPROBNPPRE",
]

# Recompute totals from the three anatomical components in BOTH sites.
# A total is only produced when all three components are present.
RECALCULATE_CALCIUM_TOTALS = True
CALCIUM_GROUPS = {
    "CT_ValvScTot": ["CT_ValScNCC", "CT_ValScRCC", "CT_ValScLCC"],
    "CT_AnnScTot": ["CT_AnnScNCC", "CT_AnnScRCC", "CT_AnnScLCC"],
    "CT_LVOTScTot": ["CT_LVOTScNCC", "CT_LVOTScRCC", "CT_LVOTScLCC"],
}

# The uploaded LMU table contains raw perimeter and area in columns whose names
# indicate derived diameters. TUM already contains derived diameters.
# LMU conversion:
#   perimeter-derived diameter = perimeter / pi
#   area-derived diameter      = 2 * sqrt(area / pi)
CONVERT_LMU_GEOMETRY_TO_DERIVED_DIAMETERS = True
PERIMETER_DERIVED_COLUMN = "CT_Peri_Deri"
AREA_DERIVED_COLUMN = "CT_Area_Deri"

# TUM LVEF is top-coded at 60 in the supplied table. To prevent values >60 from
# identifying LMU, harmonize both sites to the same ceiling.
# This is a harmonization assumption and should be documented in the methods.
CAP_LVEF_AT_60 = True
LVEF_COLUMN = "LVEFPRE"
LVEF_CAP = 60.0

# LMU ECC_INDEX is available only at ~2-decimal precision in the supplied file.
# Lost precision cannot be recreated. Therefore both sites are rounded to 2
# decimals to prevent precision itself from becoming a site signal.
ROUND_ECC_INDEX = True
ECC_INDEX_COLUMN = "ECC_INDEX"
ECC_INDEX_DECIMALS = 2

# If True, construction stops rather than silently aligning mismatched schemas.
STRICT_IDENTICAL_COLUMNS = True

# Missingness is audited, but columns are NOT automatically dropped based on a
# numeric threshold. This avoids silently removing clinically useful variables.
MISSINGNESS_WARNING_THRESHOLD = 0.40
LABEL_MISSINGNESS_GAP_WARNING = 0.20

# Audit-plot / significance-analysis configuration.
TOP_MISSINGNESS_COLUMNS = 10
SIGNIFICANCE_ALPHA_FDR = 0.05
PLOT_DPI = 300
SIGNIFICANCE_FIGSIZE = (14, 7)
MISSINGNESS_FIGSIZE = (14, 7)
BASE_FONTSIZE = 20
TITLE_FONTSIZE = 24
LABEL_FONTSIZE = 22
TICK_FONTSIZE = 18
LEGEND_FONTSIZE = 18

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"
}

ID_COLUMN_CANDIDATES = [
    "ID", "PATIENT_ID", "PATIENTID", "PatientID", "patient_id",
    "Patient_ID", "IMAGE_ID", "IMAGEID", "ImageID", "image_id",
    "SUBJECT_ID", "SUBJECTID", "SubjectID", "subject_id",
]


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def normalize_identifier(value) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    suffix = Path(text).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        text = Path(text).stem.strip()
    if re.fullmatch(r"-?\d+\.0+", text):
        text = text.split(".")[0]
    return text


def find_id_column(df: pd.DataFrame, requested_column: Optional[str]) -> str:
    if requested_column is not None:
        if requested_column not in df.columns:
            raise ValueError(
                f"Configured ID column {requested_column!r} not found. "
                f"Available: {list(df.columns)}"
            )
        return requested_column

    for candidate in ID_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate

    lower_to_original = {str(c).strip().lower(): c for c in df.columns}
    for candidate in ID_COLUMN_CANDIDATES:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]

    raise ValueError(f"Could not determine ID column. Columns: {list(df.columns)}")


def normalize_binary_label(value) -> int:
    value_str = str(value).strip().lower()
    mapping = {
        "no event": 0, "no_event": 0, "noevent": 0, "0": 0,
        "pacer": 1, "pacemaker": 1, "1": 1,
    }
    if value_str not in mapping:
        raise ValueError(f"Unsupported LABEL value: {value!r}")
    return mapping[value_str]


def validate_required_columns(df: pd.DataFrame, dataset_name: str) -> None:
    required = [LABEL_COLUMN, SEX_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} missing required columns: {missing}")

    for column in required:
        if df[column].isna().any():
            raise ValueError(
                f"{dataset_name} has {int(df[column].isna().sum())} missing "
                f"values in required column {column!r}."
            )


def validate_unique_ids(df: pd.DataFrame, id_column: str, dataset_name: str) -> None:
    normalized = df[id_column].map(normalize_identifier)
    if normalized.isna().any():
        raise ValueError(f"{dataset_name} contains unusable/missing IDs.")
    duplicated = normalized.duplicated(keep=False)
    if duplicated.any():
        ids = normalized.loc[duplicated].drop_duplicates().head(20).tolist()
        raise ValueError(f"{dataset_name} contains duplicate IDs: {ids}")


def ensure_no_cross_site_id_overlap(
    tum_df: pd.DataFrame,
    lmu_df: pd.DataFrame,
    tum_id_column: str,
    lmu_id_column: str,
) -> None:
    tum_ids = set(tum_df[tum_id_column].map(normalize_identifier).dropna())
    lmu_ids = set(lmu_df[lmu_id_column].map(normalize_identifier).dropna())
    overlap = sorted(tum_ids.intersection(lmu_ids))
    if overlap:
        raise ValueError(
            f"TUM and LMU contain {len(overlap)} overlapping normalized IDs. "
            f"First examples: {overlap[:20]}"
        )


def make_output_dirs() -> None:
    for path in [SPLIT_OUTPUT_ROOT, DATA_SIZE_OUTPUT_ROOT, AUDIT_OUTPUT_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def select_export_columns(
    df: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Return exactly the five requested modelling columns, in the requested order.

    Cohort construction and split assignment are performed on the full internal
    dataframe first. Projection to EXPORT_COLUMNS happens only when files are
    written, so row membership and split membership remain unchanged.
    """
    missing = [column for column in EXPORT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name}: cannot export 5-feature dataset because these "
            f"columns are missing: {missing}. Available columns: {list(df.columns)}"
        )

    output = df.loc[:, EXPORT_COLUMNS].copy()

    if list(output.columns) != EXPORT_COLUMNS:
        raise AssertionError(
            f"{dataset_name}: exported column order does not match EXPORT_COLUMNS."
        )
    if len(output) != len(df):
        raise AssertionError(
            f"{dataset_name}: export changed the number of rows unexpectedly."
        )

    return output


# =============================================================================
# HARMONIZATION
# =============================================================================

def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        raise ValueError(f"Required harmonization column {column!r} is missing.")
    return pd.to_numeric(df[column], errors="coerce")


def recalculate_calcium_totals(
    df: pd.DataFrame,
    dataset_name: str,
    change_log: List[Dict[str, object]],
) -> pd.DataFrame:
    output = df.copy()

    for total_column, components in CALCIUM_GROUPS.items():
        missing_components = [c for c in components if c not in output.columns]
        if missing_components:
            raise ValueError(
                f"{dataset_name}: cannot calculate {total_column}; "
                f"missing {missing_components}."
            )

        component_frame = output[components].apply(pd.to_numeric, errors="coerce")
        recalculated = component_frame.sum(axis=1, min_count=len(components))

        old = (
            pd.to_numeric(output[total_column], errors="coerce")
            if total_column in output.columns
            else pd.Series(np.nan, index=output.index)
        )

        comparable = old.notna() & recalculated.notna()
        changed = comparable & (~np.isclose(old, recalculated, equal_nan=True))
        large_difference = comparable & ((old - recalculated).abs() > 5)

        change_log.append({
            "dataset": dataset_name,
            "operation": "recalculate_calcium_total",
            "column": total_column,
            "rows": len(output),
            "old_nonmissing": int(old.notna().sum()),
            "new_nonmissing": int(recalculated.notna().sum()),
            "changed_comparable_rows": int(changed.sum()),
            "old_vs_new_abs_diff_gt_5": int(large_difference.sum()),
            "detail": "+".join(components),
        })

        output[total_column] = recalculated

    return output


def convert_lmu_geometry(
    lmu_df: pd.DataFrame,
    change_log: List[Dict[str, object]],
) -> pd.DataFrame:
    output = lmu_df.copy()

    perimeter = _numeric(output, PERIMETER_DERIVED_COLUMN)
    area = _numeric(output, AREA_DERIVED_COLUMN)

    new_perimeter_derived = perimeter / math.pi
    new_area_derived = 2.0 * np.sqrt(area / math.pi)

    change_log.append({
        "dataset": "LMU",
        "operation": "convert_raw_perimeter_to_derived_diameter",
        "column": PERIMETER_DERIVED_COLUMN,
        "rows": len(output),
        "old_nonmissing": int(perimeter.notna().sum()),
        "new_nonmissing": int(new_perimeter_derived.notna().sum()),
        "changed_comparable_rows": int((perimeter.notna()).sum()),
        "old_vs_new_abs_diff_gt_5": int(
            ((perimeter - new_perimeter_derived).abs() > 5).fillna(False).sum()
        ),
        "detail": "new = old / pi",
    })

    change_log.append({
        "dataset": "LMU",
        "operation": "convert_raw_area_to_derived_diameter",
        "column": AREA_DERIVED_COLUMN,
        "rows": len(output),
        "old_nonmissing": int(area.notna().sum()),
        "new_nonmissing": int(new_area_derived.notna().sum()),
        "changed_comparable_rows": int((area.notna()).sum()),
        "old_vs_new_abs_diff_gt_5": int(
            ((area - new_area_derived).abs() > 5).fillna(False).sum()
        ),
        "detail": "new = 2*sqrt(old/pi)",
    })

    output[PERIMETER_DERIVED_COLUMN] = new_perimeter_derived
    output[AREA_DERIVED_COLUMN] = new_area_derived
    return output


def cap_lvef(
    df: pd.DataFrame,
    dataset_name: str,
    change_log: List[Dict[str, object]],
) -> pd.DataFrame:
    output = df.copy()
    values = _numeric(output, LVEF_COLUMN)
    changed = values > LVEF_CAP
    output[LVEF_COLUMN] = values.clip(upper=LVEF_CAP)

    change_log.append({
        "dataset": dataset_name,
        "operation": "cap_lvef",
        "column": LVEF_COLUMN,
        "rows": len(output),
        "old_nonmissing": int(values.notna().sum()),
        "new_nonmissing": int(output[LVEF_COLUMN].notna().sum()),
        "changed_comparable_rows": int(changed.fillna(False).sum()),
        "old_vs_new_abs_diff_gt_5": int(((values - LVEF_CAP) > 5).fillna(False).sum()),
        "detail": f"values > {LVEF_CAP:g} replaced by {LVEF_CAP:g}",
    })
    return output


def round_ecc_index(
    df: pd.DataFrame,
    dataset_name: str,
    change_log: List[Dict[str, object]],
) -> pd.DataFrame:
    output = df.copy()
    values = _numeric(output, ECC_INDEX_COLUMN)
    rounded = values.round(ECC_INDEX_DECIMALS)
    changed = values.notna() & (~np.isclose(values, rounded, equal_nan=True))
    output[ECC_INDEX_COLUMN] = rounded

    change_log.append({
        "dataset": dataset_name,
        "operation": "round_ecc_index",
        "column": ECC_INDEX_COLUMN,
        "rows": len(output),
        "old_nonmissing": int(values.notna().sum()),
        "new_nonmissing": int(rounded.notna().sum()),
        "changed_comparable_rows": int(changed.sum()),
        "old_vs_new_abs_diff_gt_5": 0,
        "detail": f"rounded to {ECC_INDEX_DECIMALS} decimals",
    })
    return output


def drop_problematic_features(
    df: pd.DataFrame,
    dataset_name: str,
    change_log: List[Dict[str, object]],
) -> pd.DataFrame:
    output = df.copy()
    missing = [c for c in DROP_FEATURE_COLUMNS if c not in output.columns]
    if missing:
        raise ValueError(
            f"{dataset_name}: configured DROP_FEATURE_COLUMNS are absent: {missing}. "
            "Update the configuration explicitly rather than silently continuing."
        )

    for column in DROP_FEATURE_COLUMNS:
        change_log.append({
            "dataset": dataset_name,
            "operation": "drop_feature",
            "column": column,
            "rows": len(output),
            "old_nonmissing": int(output[column].notna().sum()),
            "new_nonmissing": 0,
            "changed_comparable_rows": len(output),
            "old_vs_new_abs_diff_gt_5": 0,
            "detail": "removed from BOTH sites to preserve identical schema",
        })

    return output.drop(columns=DROP_FEATURE_COLUMNS)


def enforce_identical_schema(
    tum_df: pd.DataFrame,
    lmu_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tum_columns = list(tum_df.columns)
    lmu_columns = list(lmu_df.columns)

    if set(tum_columns) != set(lmu_columns):
        tum_only = sorted(set(tum_columns) - set(lmu_columns))
        lmu_only = sorted(set(lmu_columns) - set(tum_columns))
        message = (
            "TUM/LMU schemas differ after harmonization.\n"
            f"TUM-only columns: {tum_only}\n"
            f"LMU-only columns: {lmu_only}"
        )
        if STRICT_IDENTICAL_COLUMNS:
            raise ValueError(message)
        print("WARNING:", message)

    # Canonical order is always TUM's order.
    lmu_df = lmu_df.reindex(columns=tum_columns)
    return tum_df, lmu_df


def harmonize_base_datasets(
    tum_df: pd.DataFrame,
    lmu_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply deterministic site-harmonization before cohort filtering/splitting."""
    change_log: List[Dict[str, object]] = []

    tum = tum_df.copy()
    lmu = lmu_df.copy()

    if RECALCULATE_CALCIUM_TOTALS:
        tum = recalculate_calcium_totals(tum, "TUM", change_log)
        lmu = recalculate_calcium_totals(lmu, "LMU", change_log)

    if CONVERT_LMU_GEOMETRY_TO_DERIVED_DIAMETERS:
        lmu = convert_lmu_geometry(lmu, change_log)

    if CAP_LVEF_AT_60:
        tum = cap_lvef(tum, "TUM", change_log)
        lmu = cap_lvef(lmu, "LMU", change_log)

    if ROUND_ECC_INDEX:
        tum = round_ecc_index(tum, "TUM", change_log)
        lmu = round_ecc_index(lmu, "LMU", change_log)

    tum = drop_problematic_features(tum, "TUM", change_log)
    lmu = drop_problematic_features(lmu, "LMU", change_log)

    tum, lmu = enforce_identical_schema(tum, lmu)
    return tum, lmu, pd.DataFrame(change_log)


# =============================================================================
# IMAGE COHORT FILTERING
# =============================================================================

def collect_image_identifiers(image_root: Path) -> set[str]:
    if not image_root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_root}")

    ids: set[str] = set()
    file_count = 0
    for path in image_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            normalized = normalize_identifier(path.stem)
            if normalized is not None:
                ids.add(normalized)
                file_count += 1

    if file_count == 0:
        raise RuntimeError(f"No supported images found under {image_root}")

    print(f"Found {file_count} image files representing {len(ids)} unique IDs.")
    return ids


def clean_dataframe_using_images(
    df: pd.DataFrame,
    image_ids: set[str],
    id_column: str,
    dataset_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    normalized = df[id_column].map(normalize_identifier)
    keep = normalized.notna() & normalized.isin(image_ids)
    cleaned = df.loc[keep].copy().reset_index(drop=True)
    removed = df.loc[~keep].copy().reset_index(drop=True)

    print(
        f"{dataset_name}: original={len(df)}, kept={len(cleaned)}, "
        f"removed_without_image={len(removed)}"
    )
    return cleaned, removed


# =============================================================================
# ECG INTERVAL MATCHING
# =============================================================================

def load_interval_table(interval_path: Path, dataset_name: str) -> pd.DataFrame:
    if not interval_path.exists():
        raise FileNotFoundError(f"{dataset_name} ECG file not found: {interval_path}")

    df = pd.read_excel(interval_path)
    id_column = find_id_column(df, ID_COLUMN)
    required = [QRS_COLUMN, PQ_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} ECG file missing columns: {missing}")

    output = df[[id_column, QRS_COLUMN, PQ_COLUMN]].copy()
    output["_NORMALIZED_ID"] = output[id_column].map(normalize_identifier)
    output = output.loc[output["_NORMALIZED_ID"].notna()].copy()

    dup = output["_NORMALIZED_ID"].duplicated(keep=False)
    if dup.any():
        values = output.loc[dup, "_NORMALIZED_ID"].drop_duplicates().head(20).tolist()
        raise ValueError(f"{dataset_name} ECG file has duplicate IDs: {values}")

    output[QRS_COLUMN] = pd.to_numeric(output[QRS_COLUMN], errors="coerce")
    output[PQ_COLUMN] = pd.to_numeric(output[PQ_COLUMN], errors="coerce")
    return output[["_NORMALIZED_ID", QRS_COLUMN, PQ_COLUMN]]


def add_ecg_intervals(
    dataset_df: pd.DataFrame,
    dataset_id_column: str,
    interval_df: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    output = dataset_df.copy()
    output = output.drop(
        columns=[c for c in [QRS_COLUMN, PQ_COLUMN] if c in output.columns],
        errors="ignore",
    )
    output["_NORMALIZED_ID"] = output[dataset_id_column].map(normalize_identifier)
    output = output.merge(
        interval_df,
        how="left",
        on="_NORMALIZED_ID",
        validate="many_to_one",
    ).drop(columns="_NORMALIZED_ID")

    # Keep numeric values; do not impute here.
    output[QRS_COLUMN] = pd.to_numeric(output[QRS_COLUMN], errors="coerce")
    output[PQ_COLUMN] = pd.to_numeric(output[PQ_COLUMN], errors="coerce")

    print(
        f"{dataset_name}: QRS available={int(output[QRS_COLUMN].notna().sum())}/"
        f"{len(output)}, PQ available={int(output[PQ_COLUMN].notna().sum())}/{len(output)}"
    )
    return output


# =============================================================================
# AUDIT TABLES + PLOTS
# =============================================================================

def make_missingness_audit(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Missingness rates in the FINAL model cohort, including rates by outcome."""
    y = df[LABEL_COLUMN].map(normalize_binary_label)
    rows: List[Dict[str, object]] = []

    for column in df.columns:
        if column in {ID_COLUMN, LABEL_COLUMN}:
            continue

        missing = df[column].isna()
        label0 = y == 0
        label1 = y == 1

        rate_all = float(missing.mean())
        rate0 = float(missing[label0].mean()) if label0.any() else float("nan")
        rate1 = float(missing[label1].mean()) if label1.any() else float("nan")
        gap = (
            abs(rate1 - rate0)
            if np.isfinite(rate0) and np.isfinite(rate1)
            else float("nan")
        )

        rows.append({
            "dataset": dataset_name,
            "feature": column,
            "missing_n": int(missing.sum()),
            "missing_rate": rate_all,
            "missing_percent": 100.0 * rate_all,
            "missing_rate_no_event": rate0,
            "missing_percent_no_event": 100.0 * rate0,
            "missing_rate_pacemaker": rate1,
            "missing_percent_pacemaker": 100.0 * rate1,
            "absolute_label_missingness_gap": gap,
            "absolute_label_missingness_gap_percent": 100.0 * gap,
            "high_missingness_warning": bool(
                rate_all >= MISSINGNESS_WARNING_THRESHOLD
            ),
            "label_missingness_warning": bool(
                np.isfinite(gap) and gap >= LABEL_MISSINGNESS_GAP_WARNING
            ),
        })

    return pd.DataFrame(rows).sort_values(
        ["label_missingness_warning", "absolute_label_missingness_gap", "missing_rate"],
        ascending=[False, False, False],
    )


def make_dropped_feature_missingness_report(
    df: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """Capture outcome-specific missingness BEFORE configured columns are dropped."""
    y = df[LABEL_COLUMN].map(normalize_binary_label)
    rows: List[Dict[str, object]] = []

    for column in DROP_FEATURE_COLUMNS:
        if column not in df.columns:
            raise ValueError(
                f"{dataset_name}: dropped-feature audit cannot find {column!r}."
            )

        missing = df[column].isna()
        no_event = y == 0
        pacemaker = y == 1

        rows.append({
            "dataset": dataset_name,
            "feature": column,
            "n_total": int(len(df)),
            "overall_missing_n": int(missing.sum()),
            "overall_missing_percent": 100.0 * float(missing.mean()),
            "no_event_n": int(no_event.sum()),
            "no_event_missing_n": int(missing[no_event].sum()),
            "no_event_missing_percent": (
                100.0 * float(missing[no_event].mean())
                if no_event.any() else float("nan")
            ),
            "pacemaker_n": int(pacemaker.sum()),
            "pacemaker_missing_n": int(missing[pacemaker].sum()),
            "pacemaker_missing_percent": (
                100.0 * float(missing[pacemaker].mean())
                if pacemaker.any() else float("nan")
            ),
        })

    return pd.DataFrame(rows)


def make_numeric_site_audit(
    tum_df: pd.DataFrame,
    lmu_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    ignored = {ID_COLUMN, LABEL_COLUMN}

    for column in tum_df.columns:
        if column in ignored:
            continue

        tum_values = pd.to_numeric(tum_df[column], errors="coerce")
        lmu_values = pd.to_numeric(lmu_df[column], errors="coerce")

        t = tum_values.dropna().to_numpy(dtype=float)
        l = lmu_values.dropna().to_numpy(dtype=float)

        if len(t) == 0 or len(l) == 0:
            smd = float("nan")
        else:
            pooled_sd = (
                math.sqrt((np.var(t, ddof=1) + np.var(l, ddof=1)) / 2.0)
                if len(t) > 1 and len(l) > 1
                else 0.0
            )
            smd = (
                (float(np.mean(t)) - float(np.mean(l))) / pooled_sd
                if pooled_sd > 0
                else 0.0
            )

        rows.append({
            "feature": column,
            "tum_n": int(tum_values.notna().sum()),
            "lmu_n": int(lmu_values.notna().sum()),
            "tum_missing_rate": float(tum_values.isna().mean()),
            "lmu_missing_rate": float(lmu_values.isna().mean()),
            "tum_mean": float(np.mean(t)) if len(t) else float("nan"),
            "lmu_mean": float(np.mean(l)) if len(l) else float("nan"),
            "tum_std": float(np.std(t, ddof=1)) if len(t) > 1 else float("nan"),
            "lmu_std": float(np.std(l, ddof=1)) if len(l) > 1 else float("nan"),
            "standardized_mean_difference_tum_minus_lmu": smd,
            "absolute_smd": abs(smd) if np.isfinite(smd) else float("nan"),
        })

    return pd.DataFrame(rows).sort_values("absolute_smd", ascending=False)


def _set_audit_plot_fonts() -> None:
    plt.rcParams.update({
        "font.size": BASE_FONTSIZE,
        "axes.titlesize": TITLE_FONTSIZE,
        "axes.labelsize": LABEL_FONTSIZE,
        "xtick.labelsize": TICK_FONTSIZE,
        "ytick.labelsize": TICK_FONTSIZE,
        "legend.fontsize": LEGEND_FONTSIZE,
        "font.family": "sans-serif",
    })


def plot_top_missingness(
    df: pd.DataFrame,
    dataset_name: str,
    output_path: Path,
    top_n: int = TOP_MISSINGNESS_COLUMNS,
) -> pd.DataFrame:
    """
    Plot the top-N missing features AFTER all configured feature drops and after
    ECG columns have been attached. ID and LABEL are excluded.
    """
    feature_columns = [
        column for column in df.columns
        if column not in {ID_COLUMN, LABEL_COLUMN}
    ]

    table = pd.DataFrame({
        "feature": feature_columns,
        "missing_percent": [
            100.0 * float(df[column].isna().mean())
            for column in feature_columns
        ],
        "missing_n": [
            int(df[column].isna().sum())
            for column in feature_columns
        ],
    }).sort_values(
        ["missing_percent", "feature"],
        ascending=[False, True],
    ).head(top_n)

    # Plot from lowest to highest so the largest bar appears at the top.
    plot_table = table.sort_values("missing_percent", ascending=True)

    _set_audit_plot_fonts()
    fig, ax = plt.subplots(figsize=MISSINGNESS_FIGSIZE)
    y = np.arange(len(plot_table))
    ax.barh(
        y,
        plot_table["missing_percent"],
        color="0.65",
        edgecolor="black",
        linewidth=1.0,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(plot_table["feature"])
    ax.set_xlabel("Missing values (%)")
    ax.set_ylabel("Feature")
    ax.set_title(f"{dataset_name} Missingness — Top {min(top_n, len(table))}")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", linestyle=":", linewidth=1.0, color="0.82")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for position, value in enumerate(plot_table["missing_percent"].to_numpy()):
        ax.text(
            min(value + 1.0, 98.0),
            position,
            f"{value:.1f}%",
            va="center",
            fontsize=max(TICK_FONTSIZE - 2, 10),
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return table


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR correction with NaN-safe handling."""
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)

    if not np.any(valid):
        return q

    valid_indices = np.flatnonzero(valid)
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)

    adjusted = ranked * m / np.arange(1, m + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    original_order_adjusted = np.empty_like(adjusted)
    original_order_adjusted[order] = adjusted
    q[valid_indices] = original_order_adjusted
    return q


def run_feature_significance_analysis(
    df: pd.DataFrame,
    dataset_name: str,
    output_directory: Path,
) -> pd.DataFrame:
    """
    Reproduce the project's No Event vs Pacemaker significance pipeline:
      - numeric prediction features only (ID/LABEL excluded),
      - two-sided Mann-Whitney U per feature,
      - Benjamini-Hochberg FDR correction,
      - ranked q-value plot on a logarithmic y-axis.
    """
    y = df[LABEL_COLUMN].map(normalize_binary_label)
    rows: List[Dict[str, object]] = []

    for column in df.columns:
        if column in {ID_COLUMN, LABEL_COLUMN}:
            continue

        values = pd.to_numeric(df[column], errors="coerce")
        no_event = values.loc[y == 0].dropna().to_numpy(dtype=float)
        pacemaker = values.loc[y == 1].dropna().to_numpy(dtype=float)

        if len(no_event) == 0 or len(pacemaker) == 0:
            statistic = float("nan")
            p_value = float("nan")
            rank_biserial = float("nan")
        else:
            result = mannwhitneyu(
                no_event,
                pacemaker,
                alternative="two-sided",
                method="auto",
            )
            statistic = float(result.statistic)
            p_value = float(result.pvalue)
            rank_biserial = (
                2.0 * statistic / (len(no_event) * len(pacemaker)) - 1.0
            )

        rows.append({
            "Feature": column,
            "No_Event_N": int(len(no_event)),
            "Pacemaker_N": int(len(pacemaker)),
            "No_Event_Median": (
                float(np.median(no_event)) if len(no_event) else float("nan")
            ),
            "Pacemaker_Median": (
                float(np.median(pacemaker)) if len(pacemaker) else float("nan")
            ),
            "MWU_Statistic": statistic,
            "MWU_p": p_value,
            "Rank_Biserial_NoEvent_vs_Pacemaker": rank_biserial,
        })

    result_table = pd.DataFrame(rows)
    result_table["MWU_q"] = benjamini_hochberg(result_table["MWU_p"].to_numpy())
    result_table["Significant_FDR_0.05"] = (
        result_table["MWU_q"] <= SIGNIFICANCE_ALPHA_FDR
    )
    result_table = result_table.sort_values(
        ["MWU_q", "MWU_p", "Feature"],
        na_position="last",
    ).reset_index(drop=True)

    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "summary_all_features.csv"
    result_table.to_csv(csv_path, index=False)

    plot_table = result_table.loc[result_table["MWU_q"].notna()].copy()
    _set_audit_plot_fonts()
    fig, ax = plt.subplots(figsize=SIGNIFICANCE_FIGSIZE)

    if len(plot_table) > 0:
        ranks = np.arange(1, len(plot_table) + 1)
        q_for_plot = np.clip(
            plot_table["MWU_q"].to_numpy(dtype=float),
            1e-12,
            1.0,
        )
        ax.plot(
            ranks,
            q_for_plot,
            marker="o",
            markersize=6,
            linewidth=2.2,
            color="0.30",
        )

        significant_count = int(
            (plot_table["MWU_q"] <= SIGNIFICANCE_ALPHA_FDR).sum()
        )
        if significant_count > 0 and significant_count < len(plot_table):
            ax.axvline(
                significant_count + 0.5,
                color="0.55",
                linestyle=":",
                linewidth=2.0,
                label=f"Significant features: {significant_count}",
            )
        elif significant_count > 0:
            # All features significant; keep the annotation without a boundary.
            ax.text(
                0.98,
                0.06,
                f"Significant features: {significant_count}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
            )

        ax.set_xlim(0.5, len(plot_table) + 0.5)
    else:
        significant_count = 0
        ax.text(
            0.5,
            0.5,
            "No valid numeric features",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    ax.axhline(
        SIGNIFICANCE_ALPHA_FDR,
        color="0.15",
        linestyle="--",
        linewidth=2.2,
        label=f"FDR = {SIGNIFICANCE_ALPHA_FDR:.2f}",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Features ranked by FDR-adjusted p-value")
    ax.set_ylabel("FDR-adjusted p-value (q)")
    ax.set_title(f"{dataset_name} Feature Significance (FDR)")
    ax.grid(axis="y", linestyle=":", linewidth=1.0, color="0.80")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        output_directory / "feature_significance_fdr.png",
        dpi=PLOT_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)

    significant = result_table.loc[
        result_table["MWU_q"] <= SIGNIFICANCE_ALPHA_FDR
    ]
    print(
        f"{dataset_name}: {len(significant)} features significant after "
        f"BH-FDR <= {SIGNIFICANCE_ALPHA_FDR:.2f}."
    )
    if len(significant) > 0:
        print(
            significant[
                ["Feature", "MWU_p", "MWU_q", "No_Event_Median", "Pacemaker_Median"]
            ].to_string(index=False)
        )

    return result_table


def save_audits(
    tum_df: pd.DataFrame,
    lmu_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    change_log: pd.DataFrame,
    dropped_feature_missingness: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Save all final-cohort audits and the requested plots."""
    AUDIT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    datasets = {
        "TUM": tum_df,
        "LMU": lmu_df,
        "MERGED": merged_df,
    }

    missingness_tables = {
        name: make_missingness_audit(dataframe, name)
        for name, dataframe in datasets.items()
    }
    missingness_all = pd.concat(
        list(missingness_tables.values()),
        ignore_index=True,
    )
    site = make_numeric_site_audit(tum_df, lmu_df)

    missingness_plot_dir = AUDIT_OUTPUT_ROOT / "missingness_plots"
    top_missingness_tables: Dict[str, pd.DataFrame] = {}
    for name, dataframe in datasets.items():
        top_missingness_tables[name] = plot_top_missingness(
            dataframe,
            dataset_name=name,
            output_path=(
                missingness_plot_dir
                / f"{name.lower()}_top{TOP_MISSINGNESS_COLUMNS}_missingness.png"
            ),
        )

    significance_root = AUDIT_OUTPUT_ROOT / "significance"
    significance_tables: Dict[str, pd.DataFrame] = {}
    for name, dataframe in datasets.items():
        significance_tables[name] = run_feature_significance_analysis(
            dataframe,
            dataset_name=name,
            output_directory=significance_root / name.lower(),
        )

    with pd.ExcelWriter(AUDIT_OUTPUT_ROOT / "dataset_audit.xlsx") as writer:
        change_log.to_excel(writer, sheet_name="harmonization_log", index=False)
        dropped_feature_missingness.to_excel(
            writer,
            sheet_name="dropped_feature_missingness",
            index=False,
        )
        missingness_all.to_excel(writer, sheet_name="missingness_final", index=False)
        site.to_excel(writer, sheet_name="site_shift", index=False)
        for name, table in top_missingness_tables.items():
            table.to_excel(
                writer,
                sheet_name=f"top_missing_{name.lower()}"[:31],
                index=False,
            )

    schema = {
        "internal_columns": list(tum_df.columns),
        "n_internal_columns": len(tum_df.columns),
        "export_columns": EXPORT_COLUMNS,
        "n_export_columns": len(EXPORT_COLUMNS),
        "model_exports_contain_only_export_columns": True,
        "dropped_features": DROP_FEATURE_COLUMNS,
        "lvef_cap": LVEF_CAP if CAP_LVEF_AT_60 else None,
        "ecc_decimals": ECC_INDEX_DECIMALS if ROUND_ECC_INDEX else None,
        "calcium_totals_recalculated": RECALCULATE_CALCIUM_TOTALS,
        "lmu_geometry_converted": CONVERT_LMU_GEOMETRY_TO_DERIVED_DIAMETERS,
        "fixed_tests_require_equal_no_event_and_pacemaker_counts": True,
        "significance_test": "two-sided Mann-Whitney U",
        "multiple_testing_correction": "Benjamini-Hochberg FDR",
        "significance_alpha_fdr": SIGNIFICANCE_ALPHA_FDR,
    }
    with open(
        AUDIT_OUTPUT_ROOT / "schema_and_harmonization.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(schema, f, indent=2)

    return significance_tables


# =============================================================================
# FIXED TEST + FIVE DEVELOPMENT FOLDS
# =============================================================================

def make_stratification_key(df: pd.DataFrame) -> pd.Series:
    labels = df[LABEL_COLUMN].map(normalize_binary_label).astype(str)
    sex = df[SEX_COLUMN].astype(str).str.strip()
    return labels + "__" + sex


def validate_balanced_binary_labels(df: pd.DataFrame, dataset_name: str) -> None:
    labels = df[LABEL_COLUMN].map(normalize_binary_label)
    counts = labels.value_counts().to_dict()
    no_event = int(counts.get(0, 0))
    pacemaker = int(counts.get(1, 0))
    if no_event != pacemaker:
        raise AssertionError(
            f"{dataset_name} is not label-balanced: "
            f"No Event={no_event}, Pacemaker={pacemaker}."
        )
    if no_event == 0:
        raise AssertionError(f"{dataset_name} contains no samples from either class.")
    print(
        f"{dataset_name}: balanced test verified — "
        f"{no_event} No Event + {pacemaker} Pacemaker."
    )


def _allocate_test_counts_within_label(
    stratum_sizes: Dict[str, int],
    target_total: int,
    n_folds: int,
) -> Dict[str, int]:
    """
    Allocate an exact label-specific test total across SEX strata as close as
    possible to the source sex distribution while leaving >= n_folds development
    samples in every non-empty LABEL×SEX stratum.
    """
    if target_total < 0:
        raise ValueError("target_total must be non-negative.")

    capacities = {
        key: max(0, size - n_folds)
        for key, size in stratum_sizes.items()
    }
    if sum(capacities.values()) < target_total:
        raise ValueError(
            f"Cannot allocate {target_total} test samples while leaving "
            f"{n_folds} development samples per stratum. "
            f"Sizes={stratum_sizes}, capacities={capacities}"
        )

    label_total = sum(stratum_sizes.values())
    desired = {
        key: (target_total * size / label_total if label_total else 0.0)
        for key, size in stratum_sizes.items()
    }
    allocated = {key: 0 for key in stratum_sizes}

    # If feasible, ensure every SEX stratum with test capacity is represented.
    eligible = [key for key, cap in capacities.items() if cap > 0]
    if target_total >= len(eligible):
        for key in eligible:
            allocated[key] = 1

    while sum(allocated.values()) < target_total:
        candidates = [
            key for key in stratum_sizes
            if allocated[key] < capacities[key]
        ]
        if not candidates:
            raise RuntimeError("Balanced test allocation exhausted all capacities.")

        # Largest remaining quota first. Stable key tie-break keeps reproducibility.
        key = max(
            candidates,
            key=lambda item: (desired[item] - allocated[item], str(item)),
        )
        allocated[key] += 1

    return allocated


def validate_partition_indices(
    original_row_count: int,
    test_indices: np.ndarray,
    fold_indices: Sequence[np.ndarray],
    dataset_name: str,
) -> None:
    all_indices = np.concatenate(
        [np.asarray(test_indices), *[np.asarray(x) for x in fold_indices]]
    )
    if len(all_indices) != original_row_count:
        raise AssertionError(f"{dataset_name}: partition row count mismatch.")
    unique, counts = np.unique(all_indices, return_counts=True)
    if len(unique) != original_row_count or np.any(counts != 1):
        raise AssertionError(f"{dataset_name}: partition has overlap or missing rows.")
    if not np.array_equal(np.sort(unique), np.arange(original_row_count)):
        raise AssertionError(f"{dataset_name}: partition contains invalid row indices.")


def create_fixed_test_and_folds(
    df: pd.DataFrame,
    dataset_name: str,
    random_seed: int,
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    """
    Create an EXACTLY label-balanced fixed test set, then five development folds.

    Test-set rules:
      - #No Event == #Pacemaker exactly.
      - Size stays as close as feasible to TEST_FRACTION.
      - Within each label, SEX proportions are preserved as closely as possible.
      - At least N_FOLDS samples remain in every LABEL×SEX stratum so each
        development fold receives at least one sample from every stratum.
    """
    work = df.copy().reset_index(drop=True)
    work["_BINARY_LABEL"] = work[LABEL_COLUMN].map(normalize_binary_label)
    work["_SEX_KEY"] = work[SEX_COLUMN].astype(str).str.strip()
    rng = np.random.default_rng(random_seed)

    label_counts = work["_BINARY_LABEL"].value_counts().to_dict()
    if set(label_counts) != {0, 1}:
        raise ValueError(
            f"{dataset_name}: both binary labels are required. Counts={label_counts}"
        )

    stratum_sizes_by_label: Dict[int, Dict[str, int]] = {}
    capacities_by_label: Dict[int, int] = {}
    ideal_test_by_label: Dict[int, int] = {}

    for label in [0, 1]:
        label_df = work.loc[work["_BINARY_LABEL"] == label]
        sizes = {
            str(sex): int(count)
            for sex, count in label_df["_SEX_KEY"].value_counts(sort=False).items()
        }
        stratum_sizes_by_label[label] = sizes
        capacities_by_label[label] = sum(
            max(0, size - N_FOLDS)
            for size in sizes.values()
        )
        ideal_test_by_label[label] = int(
            round(int(label_counts[label]) * TEST_FRACTION)
        )

    test_per_label = min(
        ideal_test_by_label[0],
        ideal_test_by_label[1],
        capacities_by_label[0],
        capacities_by_label[1],
    )
    if test_per_label < 1:
        raise ValueError(
            f"{dataset_name}: cannot create a non-empty balanced test set. "
            f"Label counts={label_counts}, capacities={capacities_by_label}."
        )

    test_indices: List[int] = []
    fold_indices: List[List[int]] = [[] for _ in range(N_FOLDS)]

    for label in [0, 1]:
        test_counts = _allocate_test_counts_within_label(
            stratum_sizes_by_label[label],
            target_total=test_per_label,
            n_folds=N_FOLDS,
        )

        for sex_key in sorted(stratum_sizes_by_label[label]):
            stratum = work.loc[
                (work["_BINARY_LABEL"] == label)
                & (work["_SEX_KEY"] == sex_key)
            ]
            indices = stratum.index.to_numpy(dtype=np.int64).copy()
            rng.shuffle(indices)

            test_count = test_counts[str(sex_key)]
            test_part = indices[:test_count]
            dev = indices[test_count:].copy()
            rng.shuffle(dev)

            if len(dev) < N_FOLDS:
                raise AssertionError(
                    f"{dataset_name}: LABEL={label}, SEX={sex_key} leaves only "
                    f"{len(dev)} development samples for {N_FOLDS} folds."
                )

            parts = np.array_split(dev, N_FOLDS)
            if any(len(part) == 0 for part in parts):
                raise AssertionError(
                    f"{dataset_name}: empty development fold in "
                    f"LABEL={label}, SEX={sex_key}."
                )

            test_indices.extend(test_part.tolist())
            for fold_i, part in enumerate(parts):
                fold_indices[fold_i].extend(part.tolist())

            print(
                f"{dataset_name} LABEL={label}, SEX={sex_key}: "
                f"total={len(indices)}, test={test_count}, "
                f"folds={[len(part) for part in parts]}"
            )

    test_array = np.asarray(test_indices, dtype=np.int64)
    fold_arrays = [np.asarray(x, dtype=np.int64) for x in fold_indices]
    rng.shuffle(test_array)
    for array in fold_arrays:
        rng.shuffle(array)

    validate_partition_indices(
        len(work),
        test_array,
        fold_arrays,
        dataset_name,
    )

    drop_internal = ["_BINARY_LABEL", "_SEX_KEY"]
    test_df = (
        work.iloc[test_array]
        .drop(columns=drop_internal)
        .reset_index(drop=True)
    )
    folds = [
        work.iloc[array].drop(columns=drop_internal).reset_index(drop=True)
        for array in fold_arrays
    ]

    validate_balanced_binary_labels(test_df, f"{dataset_name} test")
    return test_df, folds


def save_split_collection(
    output_directory: Path,
    test_df: pd.DataFrame,
    folds: Sequence[pd.DataFrame],
) -> None:
    """
    Save split files with exactly EXPORT_COLUMNS.

    The input dataframes still contain LABEL/SEX/etc. internally because those
    columns are required to reproduce the original stratification and balancing.
    """
    output_directory.mkdir(parents=True, exist_ok=True)

    test_export = select_export_columns(test_df, "test split")
    test_export.to_excel(output_directory / "test.xlsx", index=False)

    for i, fold in enumerate(folds, start=1):
        fold_export = select_export_columns(fold, f"fold{i}")
        fold_export.to_excel(output_directory / f"fold{i}.xlsx", index=False)


def concatenate_matching_splits(
    tum_test: pd.DataFrame,
    tum_folds: Sequence[pd.DataFrame],
    lmu_test: pd.DataFrame,
    lmu_folds: Sequence[pd.DataFrame],
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    merged_test = pd.concat([tum_test, lmu_test], ignore_index=True, sort=False)
    merged_folds = [
        pd.concat([tf, lf], ignore_index=True, sort=False)
        for tf, lf in zip(tum_folds, lmu_folds)
    ]
    return merged_test, merged_folds


def verify_merged_split(
    tum_df: pd.DataFrame,
    lmu_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    split_name: str,
) -> None:
    expected = pd.concat([tum_df, lmu_df], ignore_index=True, sort=False)
    if list(tum_df.columns) != list(lmu_df.columns) or list(merged_df.columns) != list(tum_df.columns):
        raise AssertionError(f"{split_name}: column schema/order mismatch.")
    pd.testing.assert_frame_equal(
        merged_df.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
        check_like=False,
    )
    if len(merged_df) != len(tum_df) + len(lmu_df):
        raise AssertionError(f"{split_name}: merged sample count is not TUM + LMU.")
    print(f"{split_name}: verified exact TUM + LMU ({len(merged_df)} samples).")


# =============================================================================
# DATA-SIZE EXPERIMENTS
# =============================================================================

def calculate_balanced_subset_size(total_dev: int, percentage: int, n_folds: int) -> int:
    requested = total_dev * percentage / 100.0
    multiple = 2 * n_folds
    lower = max((int(requested) // multiple) * multiple, multiple)
    upper = max(lower + multiple, multiple)
    return lower if abs(lower - requested) <= abs(upper - requested) else upper


def sample_balanced_development_subset(
    folds: Sequence[pd.DataFrame],
    percentage: int,
    random_seed: int,
    dataset_name: str,
) -> List[pd.DataFrame]:
    dev = pd.concat(folds, ignore_index=True, sort=False).copy()
    dev["_BINARY_LABEL"] = dev[LABEL_COLUMN].map(normalize_binary_label)

    target_total = calculate_balanced_subset_size(len(dev), percentage, N_FOLDS)
    n_per_label = target_total // 2
    n_per_label_per_fold = n_per_label // N_FOLDS

    class0 = dev.loc[dev["_BINARY_LABEL"] == 0].copy()
    class1 = dev.loc[dev["_BINARY_LABEL"] == 1].copy()
    available = min(len(class0), len(class1))
    if n_per_label > available:
        raise ValueError(
            f"{dataset_name} {percentage}% needs {n_per_label}/class, only {available} available."
        )

    sampled0 = class0.sample(
        n=n_per_label, replace=False, random_state=random_seed + percentage * 100 + 1
    ).sample(frac=1, random_state=random_seed + percentage * 1000 + 10).reset_index(drop=True)
    sampled1 = class1.sample(
        n=n_per_label, replace=False, random_state=random_seed + percentage * 100 + 2
    ).sample(frac=1, random_state=random_seed + percentage * 1000 + 20).reset_index(drop=True)

    output: List[pd.DataFrame] = []
    for fold_index in range(N_FOLDS):
        start = fold_index * n_per_label_per_fold
        end = start + n_per_label_per_fold
        fold = pd.concat(
            [sampled0.iloc[start:end], sampled1.iloc[start:end]],
            ignore_index=True,
        ).drop(columns="_BINARY_LABEL")
        fold = fold.sample(
            frac=1,
            random_state=random_seed + percentage * 10000 + fold_index,
        ).reset_index(drop=True)
        output.append(fold)

    # Strict balance/equal-size checks.
    sizes = [len(x) for x in output]
    if len(set(sizes)) != 1:
        raise AssertionError(f"{dataset_name} {percentage}% folds not equal sized: {sizes}")
    for i, fold in enumerate(output, start=1):
        counts = fold[LABEL_COLUMN].map(normalize_binary_label).value_counts().to_dict()
        if counts.get(0, 0) != counts.get(1, 0):
            raise AssertionError(f"{dataset_name} {percentage}% fold{i} not label-balanced.")

    print(
        f"{dataset_name} {percentage}%: selected {sum(sizes)}/{len(dev)} dev samples, "
        f"fold sizes={sizes}"
    )
    return output


def create_all_data_size_experiment_datasets(
    tum_test: pd.DataFrame,
    tum_folds: Sequence[pd.DataFrame],
    lmu_test: pd.DataFrame,
    lmu_folds: Sequence[pd.DataFrame],
) -> None:
    for percentage in DATA_SIZE_PERCENTAGES:
        tum_pct = sample_balanced_development_subset(
            tum_folds, percentage, RANDOM_SEED, "TUM"
        )
        lmu_pct = sample_balanced_development_subset(
            lmu_folds, percentage, RANDOM_SEED + 1, "LMU"
        )

        merged_test = pd.concat([tum_test, lmu_test], ignore_index=True, sort=False)
        merged_pct = [
            pd.concat([tum_pct[i], lmu_pct[i]], ignore_index=True, sort=False)
            for i in range(N_FOLDS)
        ]

        root = DATA_SIZE_OUTPUT_ROOT / f"{percentage}_percent"
        save_split_collection(root / "tum", tum_test, tum_pct)
        save_split_collection(root / "lmu", lmu_test, lmu_pct)
        save_split_collection(root / "merged", merged_test, merged_pct)

        validate_balanced_binary_labels(tum_test, f"{percentage}% TUM test")
        validate_balanced_binary_labels(lmu_test, f"{percentage}% LMU test")
        validate_balanced_binary_labels(merged_test, f"{percentage}% MERGED test")
        verify_merged_split(tum_test, lmu_test, merged_test, f"{percentage}% merged test")
        for i in range(N_FOLDS):
            verify_merged_split(
                tum_pct[i], lmu_pct[i], merged_pct[i],
                f"{percentage}% merged fold{i + 1}",
            )


# =============================================================================
# REPORTING
# =============================================================================

def print_dataset_summary(df: pd.DataFrame, name: str) -> None:
    labels = df[LABEL_COLUMN].map(normalize_binary_label)
    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)
    print(f"Samples: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"No event: {int((labels == 0).sum())}")
    print(f"Pacemaker: {int((labels == 1).sum())}")
    print(f"QRS available: {int(df[QRS_COLUMN].notna().sum()) if QRS_COLUMN in df else 0}")
    print(f"PQ available: {int(df[PQ_COLUMN].notna().sum()) if PQ_COLUMN in df else 0}")


def write_construction_summary(
    tum_df: pd.DataFrame,
    lmu_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    tum_test: pd.DataFrame,
    lmu_test: pd.DataFrame,
    merged_test: pd.DataFrame,
    dropped_feature_missingness: pd.DataFrame,
    significance_tables: Dict[str, pd.DataFrame],
) -> None:
    def _label_counts(dataframe: pd.DataFrame) -> Tuple[int, int]:
        labels = dataframe[LABEL_COLUMN].map(normalize_binary_label)
        return int((labels == 0).sum()), int((labels == 1).sum())

    tum_test_no, tum_test_pacer = _label_counts(tum_test)
    lmu_test_no, lmu_test_pacer = _label_counts(lmu_test)
    merged_test_no, merged_test_pacer = _label_counts(merged_test)

    lines = [
        "FINAL DATASET CONSTRUCTION SUMMARY",
        "=" * 72,
        "",
        "Harmonization:",
        f"- Dropped from BOTH sites: {', '.join(DROP_FEATURE_COLUMNS)}",
        "- Recalculated CT_ValvScTot, CT_AnnScTot, CT_LVOTScTot from components.",
        "- Converted LMU CT_Peri_Deri = perimeter/pi.",
        "- Converted LMU CT_Area_Deri = 2*sqrt(area/pi).",
        f"- LVEF capped at {LVEF_CAP:g} in both sites: {CAP_LVEF_AT_60}.",
        f"- ECC_INDEX rounded to {ECC_INDEX_DECIMALS} decimals in both sites: {ROUND_ECC_INDEX}.",
        "- Same image-available cohort used for tabular, CT-only and combined models.",
        "- TUM and LMU final columns/order are required to be identical.",
        "- Every merged dataset is required to equal exact TUM + LMU concatenation.",
        "",
        "WHY DM / CABGPRE / NTPROBNPPRE WERE DROPPED:",
        "- These percentages are measured in the ORIGINAL source tables before dropping.",
    ]

    lmu_drop = dropped_feature_missingness.loc[
        dropped_feature_missingness["dataset"] == "LMU"
    ]
    for _, row in lmu_drop.iterrows():
        lines.append(
            f"- LMU {row['feature']}: missing in "
            f"{row['no_event_missing_percent']:.1f}% of No Event and "
            f"{row['pacemaker_missing_percent']:.1f}% of Pacemaker patients "
            f"(overall {row['overall_missing_percent']:.1f}%)."
        )

    lines += [
        "",
        "DIAGNOSTIC LOGISTIC-REGRESSION SANITY CHECK (NOT A CLINICAL MODEL):",
        "- scikit-learn LogisticRegression(max_iter=5000).",
        "- Default L2 regularization, C=1.0, lbfgs solver.",
        "- Shuffled StratifiedKFold(n_splits=5, random_state=42) with out-of-fold probabilities.",
        "- It was used only to test whether values/missingness patterns could predict the label.",
        "",
        "Final cohorts:",
        f"- TUM: {len(tum_df)}",
        f"- LMU: {len(lmu_df)}",
        f"- Merged: {len(merged_df)} = {len(tum_df)} + {len(lmu_df)}",
        "",
        "Fixed tests — EXACT label balance enforced:",
        f"- TUM test: {len(tum_test)} = {tum_test_no} No Event + {tum_test_pacer} Pacemaker",
        f"- LMU test: {len(lmu_test)} = {lmu_test_no} No Event + {lmu_test_pacer} Pacemaker",
        f"- Merged test: {len(merged_test)} = {merged_test_no} No Event + {merged_test_pacer} Pacemaker",
        "",
        "Exported modelling files:",
        f"- Output root: {OUTPUT_ROOT}",
        f"- Every cleaned/test/fold/data-size modelling workbook contains exactly {len(EXPORT_COLUMNS)} columns.",
        f"- Export columns: {', '.join(EXPORT_COLUMNS)}",
        "- Split membership is still created from the full internal dataframe before the five-column projection.",
        "",
        "Final missingness plots:",
        f"- Top {TOP_MISSINGNESS_COLUMNS} missing features are plotted AFTER configured feature drops for TUM, LMU and Merged.",
        "",
        "No Event vs Pacemaker significance analysis:",
        "- Two-sided Mann-Whitney U test for every numeric model feature.",
        f"- Benjamini-Hochberg FDR correction at q <= {SIGNIFICANCE_ALPHA_FDR:.2f}.",
    ]

    for name in ["TUM", "LMU", "MERGED"]:
        table = significance_tables[name]
        n_valid = int(table["MWU_q"].notna().sum())
        n_sig = int((table["MWU_q"] <= SIGNIFICANCE_ALPHA_FDR).sum())
        lines.append(
            f"- {name}: {n_sig}/{n_valid} tested features significant after FDR correction."
        )

    lines += [
        "",
        f"Export columns ({len(EXPORT_COLUMNS)}):",
        *[f"  {column}" for column in EXPORT_COLUMNS],
        "",
        f"Internal construction columns ({len(tum_df.columns)}):",
        *[f"  {column}" for column in tum_df.columns],
    ]

    (AUDIT_OUTPUT_ROOT / "construction_summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    make_output_dirs()

    if not TUM_EXCEL_PATH.exists() or not LMU_EXCEL_PATH.exists():
        raise FileNotFoundError(
            f"Input files not found: TUM={TUM_EXCEL_PATH}, LMU={LMU_EXCEL_PATH}"
        )

    tum_raw = pd.read_excel(TUM_EXCEL_PATH)
    lmu_raw = pd.read_excel(LMU_EXCEL_PATH)

    validate_required_columns(tum_raw, "TUM raw")
    validate_required_columns(lmu_raw, "LMU raw")

    tum_id = find_id_column(tum_raw, ID_COLUMN)
    lmu_id = find_id_column(lmu_raw, ID_COLUMN)
    validate_unique_ids(tum_raw, tum_id, "TUM raw")
    validate_unique_ids(lmu_raw, lmu_id, "LMU raw")
    ensure_no_cross_site_id_overlap(tum_raw, lmu_raw, tum_id, lmu_id)

    # Capture the missingness that motivated feature removal BEFORE those
    # columns disappear from the harmonized model tables.
    dropped_feature_missingness = pd.concat(
        [
            make_dropped_feature_missingness_report(tum_raw, "TUM"),
            make_dropped_feature_missingness_report(lmu_raw, "LMU"),
        ],
        ignore_index=True,
    )

    # -------------------------------------------------------------------------
    # Harmonize BEFORE splitting.
    # -------------------------------------------------------------------------
    tum_h, lmu_h, change_log = harmonize_base_datasets(tum_raw, lmu_raw)
    tum_h, lmu_h = enforce_identical_schema(tum_h, lmu_h)

    # -------------------------------------------------------------------------
    # Keep only patients with images -> identical cohort across modalities.
    # -------------------------------------------------------------------------
    image_ids = collect_image_identifiers(IMAGE_ROOT)
    tum_clean, tum_removed = clean_dataframe_using_images(tum_h, image_ids, tum_id, "TUM")
    lmu_clean, lmu_removed = clean_dataframe_using_images(lmu_h, image_ids, lmu_id, "LMU")

    removed_dir = SPLIT_OUTPUT_ROOT / "removed_rows"
    removed_dir.mkdir(parents=True, exist_ok=True)
    tum_removed.to_excel(removed_dir / "tum_removed_missing_images.xlsx", index=False)
    lmu_removed.to_excel(removed_dir / "lmu_removed_missing_images.xlsx", index=False)

    # -------------------------------------------------------------------------
    # ECG intervals.
    # -------------------------------------------------------------------------
    tum_intervals = load_interval_table(TUM_ECG_INTERVAL_PATH, "TUM")
    lmu_intervals = load_interval_table(LMU_ECG_INTERVAL_PATH, "LMU")
    tum_clean = add_ecg_intervals(tum_clean, tum_id, tum_intervals, "TUM")
    lmu_clean = add_ecg_intervals(lmu_clean, lmu_id, lmu_intervals, "LMU")

    # Adding ECG columns may change column order; enforce one canonical order.
    tum_clean, lmu_clean = enforce_identical_schema(tum_clean, lmu_clean)
    validate_unique_ids(tum_clean, tum_id, "TUM cleaned")
    validate_unique_ids(lmu_clean, lmu_id, "LMU cleaned")
    ensure_no_cross_site_id_overlap(tum_clean, lmu_clean, tum_id, lmu_id)

    # -------------------------------------------------------------------------
    # Save final site datasets + exact merged dataset.
    # -------------------------------------------------------------------------
    merged = pd.concat([tum_clean, lmu_clean], ignore_index=True, sort=False)
    if list(merged.columns) != list(tum_clean.columns):
        raise AssertionError("Merged dataset column order changed unexpectedly.")
    if len(merged) != len(tum_clean) + len(lmu_clean):
        raise AssertionError("Merged sample count != TUM + LMU.")

    # Export only the requested five modelling features. The full dataframes
    # remain in memory for auditing and for reproducing the exact same splits.
    tum_export = select_export_columns(tum_clean, "TUM cleaned")
    lmu_export = select_export_columns(lmu_clean, "LMU cleaned")
    merged_export = select_export_columns(merged, "Merged cleaned")

    tum_export.to_excel(TUM_CLEANED_OUTPUT_PATH, index=False)
    lmu_export.to_excel(LMU_CLEANED_OUTPUT_PATH, index=False)
    merged_export.to_excel(ENTIRE_OUTPUT_PATH, index=False)

    # Audit FINAL model cohort, not just raw files. Missingness plots and
    # significance plots are generated for TUM, LMU and exact merged data.
    significance_tables = save_audits(
        tum_clean,
        lmu_clean,
        merged,
        change_log,
        dropped_feature_missingness,
    )

    # -------------------------------------------------------------------------
    # Fixed test + development folds.
    # -------------------------------------------------------------------------
    tum_test, tum_folds = create_fixed_test_and_folds(tum_clean, "TUM", RANDOM_SEED)
    lmu_test, lmu_folds = create_fixed_test_and_folds(lmu_clean, "LMU", RANDOM_SEED + 1)
    merged_test, merged_folds = concatenate_matching_splits(
        tum_test, tum_folds, lmu_test, lmu_folds
    )

    validate_balanced_binary_labels(tum_test, "TUM test")
    validate_balanced_binary_labels(lmu_test, "LMU test")
    validate_balanced_binary_labels(merged_test, "MERGED test")

    verify_merged_split(tum_test, lmu_test, merged_test, "merged test")
    for i in range(N_FOLDS):
        verify_merged_split(tum_folds[i], lmu_folds[i], merged_folds[i], f"merged fold{i+1}")

    save_split_collection(SPLIT_OUTPUT_ROOT / "tum", tum_test, tum_folds)
    save_split_collection(SPLIT_OUTPUT_ROOT / "lmu", lmu_test, lmu_folds)
    save_split_collection(SPLIT_OUTPUT_ROOT / "merged", merged_test, merged_folds)

    # -------------------------------------------------------------------------
    # Data-size experiments, always using the SAME original fixed test sets.
    # -------------------------------------------------------------------------
    create_all_data_size_experiment_datasets(tum_test, tum_folds, lmu_test, lmu_folds)

    write_construction_summary(
        tum_clean,
        lmu_clean,
        merged,
        tum_test,
        lmu_test,
        merged_test,
        dropped_feature_missingness,
        significance_tables,
    )

    print_dataset_summary(tum_clean, "FINAL TUM")
    print_dataset_summary(lmu_clean, "FINAL LMU")
    print_dataset_summary(merged, "FINAL MERGED")

    print("\nConstruction completed successfully.")
    print(f"Audit workbook: {AUDIT_OUTPUT_ROOT / 'dataset_audit.xlsx'}")
    print(f"Summary: {AUDIT_OUTPUT_ROOT / 'construction_summary.txt'}")


if __name__ == "__main__":
    main()
