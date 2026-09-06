#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


# ============================================================
# CONFIGURATION
# ============================================================

TUM_EXCEL_PATH = Path(
    "/home/ubuntu/TAVI_final/dataset/construct/tum_cleaned.xlsx"
)

LMU_EXCEL_PATH = Path(
    "/home/ubuntu/TAVI_final/dataset/construct/lmu_cleaned.xlsx"
)

OUTPUT_ROOT = Path(
    "feature_significance_tum_vs_lmu"
)

ALPHA_FDR = 0.05

# A feature must have at least this many available values in each center.
MIN_SAMPLES_PER_GROUP = 2

# Columns that should never be statistically compared.
EXCLUDE_COLUMNS = {
    "ID",
    "PATIENT_ID",
    "PATIENTID",
    "PatientID",
    "patient_id",
    "Patient_ID",
    "IMAGE_ID",
    "IMAGEID",
    "ImageID",
    "image_id",
    "SUBJECT_ID",
    "SUBJECTID",
    "SubjectID",
    "subject_id",
    "LABEL",
}

SUMMARY_FILENAME = "summary_tum_vs_lmu.csv"
SKIPPED_FILENAME = "skipped_features.csv"
SIGNIFICANT_FILENAME = "significant_features.csv"
PLOT_FILENAME = "significance_plot_tum_vs_lmu.png"

PLOT_DPI = 300

# Presentation-friendly fonts.
plt.rcParams.update(
    {
        "font.size": 20,
        "font.family": "sans-serif",
        "axes.titlesize": 24,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 20,
    }
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
)


# ============================================================
# DATA LOADING
# ============================================================

def load_excel_file(
    excel_path: Path,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Load one Excel dataset and print basic information.
    """
    if not excel_path.exists():
        raise FileNotFoundError(
            f"{dataset_name} Excel file does not exist: "
            f"{excel_path}"
        )

    dataframe = pd.read_excel(
        excel_path
    )

    if dataframe.empty:
        raise RuntimeError(
            f"{dataset_name} dataset is empty: {excel_path}"
        )

    print()
    print(f"{dataset_name} dataset")
    print("-" * 70)
    print(f"Path:       {excel_path}")
    print(f"Rows:       {len(dataframe)}")
    print(f"Columns:    {len(dataframe.columns)}")

    return dataframe


# ============================================================
# FEATURE DISCOVERY
# ============================================================

def convert_column_to_numeric(
    series: pd.Series,
):
    """
    Convert a column to numeric.

    Returns:
        converted series
        number of non-missing numeric values
    """
    converted = pd.to_numeric(
        series,
        errors="coerce",
    )

    valid_count = int(
        converted.notna().sum()
    )

    return converted, valid_count


def find_shared_numeric_features(
    tum_df: pd.DataFrame,
    lmu_df: pd.DataFrame,
):
    """
    Find feature columns available in both TUM and LMU that can be
    interpreted numerically.

    A feature does not need to have the same pandas dtype in both files.
    For example, one Excel file may load an integer column as object because
    it contains some missing or malformed values.
    """
    shared_columns = [
        column
        for column in tum_df.columns
        if column in lmu_df.columns
    ]

    numeric_features = []
    excluded_features = []

    for column in shared_columns:
        if column in EXCLUDE_COLUMNS:
            excluded_features.append(
                {
                    "Feature": column,
                    "Reason": "Explicitly excluded",
                }
            )
            continue

        tum_numeric, tum_count = convert_column_to_numeric(
            tum_df[column]
        )

        lmu_numeric, lmu_count = convert_column_to_numeric(
            lmu_df[column]
        )

        # A column is considered usable if both datasets contain at least
        # one value that can be interpreted numerically.
        if tum_count == 0 or lmu_count == 0:
            excluded_features.append(
                {
                    "Feature": column,
                    "Reason": (
                        "No numeric values in one or both datasets"
                    ),
                }
            )
            continue

        numeric_features.append(
            column
        )

    tum_only_columns = sorted(
        set(tum_df.columns)
        - set(lmu_df.columns)
    )

    lmu_only_columns = sorted(
        set(lmu_df.columns)
        - set(tum_df.columns)
    )

    print()
    print("Feature discovery")
    print("-" * 70)
    print(
        f"Shared columns:             "
        f"{len(shared_columns)}"
    )
    print(
        f"Shared numeric features:    "
        f"{len(numeric_features)}"
    )
    print(
        f"TUM-only columns:           "
        f"{len(tum_only_columns)}"
    )
    print(
        f"LMU-only columns:           "
        f"{len(lmu_only_columns)}"
    )

    if tum_only_columns:
        print()
        print(
            "TUM-only columns excluded:"
        )
        print(
            ", ".join(
                tum_only_columns
            )
        )

    if lmu_only_columns:
        print()
        print(
            "LMU-only columns excluded:"
        )
        print(
            ", ".join(
                lmu_only_columns
            )
        )

    if not numeric_features:
        raise RuntimeError(
            "No shared numeric features were found between "
            "the TUM and LMU datasets."
        )

    return (
        numeric_features,
        excluded_features,
    )


# ============================================================
# STATISTICAL FUNCTIONS
# ============================================================

def calculate_mann_whitney(
    tum_values: pd.Series,
    lmu_values: pd.Series,
):
    """
    Run a two-sided Mann–Whitney U test comparing TUM and LMU.
    """
    tum_values = pd.to_numeric(
        tum_values,
        errors="coerce",
    ).dropna()

    lmu_values = pd.to_numeric(
        lmu_values,
        errors="coerce",
    ).dropna()

    if (
        len(tum_values) < MIN_SAMPLES_PER_GROUP
        or len(lmu_values) < MIN_SAMPLES_PER_GROUP
    ):
        return np.nan, np.nan

    try:
        statistic, p_value = mannwhitneyu(
            tum_values,
            lmu_values,
            alternative="two-sided",
            method="auto",
        )

        return (
            float(statistic),
            float(p_value),
        )

    except Exception:
        return np.nan, np.nan


def benjamini_hochberg_fdr(
    p_values: pd.Series,
) -> pd.Series:
    """
    Apply Benjamini–Hochberg FDR correction to finite p-values.

    Missing p-values remain missing.
    """
    p = pd.to_numeric(
        p_values,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    adjusted = np.full(
        len(p),
        np.nan,
        dtype=float,
    )

    valid_mask = np.isfinite(
        p
    )

    valid_indices = np.where(
        valid_mask
    )[0]

    valid_p = p[
        valid_mask
    ]

    number_of_tests = len(
        valid_p
    )

    if number_of_tests == 0:
        return pd.Series(
            adjusted,
            index=p_values.index,
        )

    order = np.argsort(
        valid_p
    )

    sorted_p = valid_p[
        order
    ]

    ranks = np.arange(
        1,
        number_of_tests + 1,
        dtype=float,
    )

    sorted_q = (
        sorted_p
        * number_of_tests
        / ranks
    )

    # Ensure monotonic adjusted p-values.
    sorted_q = np.minimum.accumulate(
        sorted_q[::-1]
    )[::-1]

    sorted_q = np.clip(
        sorted_q,
        0.0,
        1.0,
    )

    valid_q = np.empty_like(
        sorted_q
    )

    valid_q[
        order
    ] = sorted_q

    adjusted[
        valid_indices
    ] = valid_q

    return pd.Series(
        adjusted,
        index=p_values.index,
    )


# ============================================================
# EFFECT-SIZE HELPERS
# ============================================================

def calculate_rank_biserial_correlation(
    mann_whitney_statistic,
    tum_count,
    lmu_count,
):
    """
    Calculate rank-biserial correlation from the Mann–Whitney statistic.

    Positive values indicate that TUM values tend to be larger.
    Negative values indicate that LMU values tend to be larger.
    """
    denominator = (
        tum_count
        * lmu_count
    )

    if denominator == 0:
        return np.nan

    effect_size = (
        2.0
        * mann_whitney_statistic
        / denominator
        - 1.0
    )

    return float(
        effect_size
    )


# ============================================================
# FEATURE ANALYSIS
# ============================================================

def analyze_features(
    tum_df: pd.DataFrame,
    lmu_df: pd.DataFrame,
    numeric_features,
):
    """
    Compare TUM and LMU for every shared numeric feature.
    """
    tested_records = []
    skipped_records = []

    for feature in numeric_features:
        tum_values = pd.to_numeric(
            tum_df[feature],
            errors="coerce",
        ).dropna()

        lmu_values = pd.to_numeric(
            lmu_df[feature],
            errors="coerce",
        ).dropna()

        tum_count = len(
            tum_values
        )

        lmu_count = len(
            lmu_values
        )

        if (
            tum_count < MIN_SAMPLES_PER_GROUP
            or lmu_count < MIN_SAMPLES_PER_GROUP
        ):
            skipped_records.append(
                {
                    "Feature": feature,
                    "TUM_N": tum_count,
                    "LMU_N": lmu_count,
                    "Reason": (
                        "Insufficient non-missing values"
                    ),
                }
            )

            print(
                f"Skipping {feature}: "
                f"TUM n={tum_count}, "
                f"LMU n={lmu_count}"
            )

            continue

        statistic, p_value = calculate_mann_whitney(
            tum_values=tum_values,
            lmu_values=lmu_values,
        )

        if not np.isfinite(
            p_value
        ):
            skipped_records.append(
                {
                    "Feature": feature,
                    "TUM_N": tum_count,
                    "LMU_N": lmu_count,
                    "Reason": (
                        "Mann–Whitney test returned NaN"
                    ),
                }
            )

            print(
                f"Skipping {feature}: "
                "Mann–Whitney test returned NaN"
            )

            continue

        effect_size = calculate_rank_biserial_correlation(
            mann_whitney_statistic=statistic,
            tum_count=tum_count,
            lmu_count=lmu_count,
        )

        tested_records.append(
            {
                "Feature": feature,
                "TUM_N": tum_count,
                "LMU_N": lmu_count,
                "MWU_stat": statistic,
                "MWU_p": p_value,
                "TUM_Median": float(
                    tum_values.median()
                ),
                "LMU_Median": float(
                    lmu_values.median()
                ),
                "TUM_Mean": float(
                    tum_values.mean()
                ),
                "LMU_Mean": float(
                    lmu_values.mean()
                ),
                "TUM_STD": float(
                    tum_values.std()
                ),
                "LMU_STD": float(
                    lmu_values.std()
                ),
                "Rank_Biserial": effect_size,
            }
        )

    if not tested_records:
        raise RuntimeError(
            "No shared numeric feature had enough valid values "
            "in both TUM and LMU."
        )

    results_df = pd.DataFrame(
        tested_records
    )

    results_df["MWU_q"] = (
        benjamini_hochberg_fdr(
            results_df["MWU_p"]
        )
    )

    results_df = (
        results_df
        .sort_values(
            by=[
                "MWU_q",
                "MWU_p",
            ],
            ascending=[
                True,
                True,
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    skipped_df = pd.DataFrame(
        skipped_records
    )

    return (
        results_df,
        skipped_df,
    )


# ============================================================
# SIGNIFICANCE PLOT
# ============================================================

def create_significance_plot(
    results_df: pd.DataFrame,
):
    """
    Plot sorted FDR-adjusted q-values.

    Features below the horizontal FDR threshold differ significantly
    between TUM and LMU.
    """
    valid_results = results_df.loc[
        np.isfinite(
            results_df["MWU_q"]
        )
    ].copy()

    if valid_results.empty:
        print(
            "WARNING: no finite q-values were available. "
            "The significance plot was not created."
        )

        return None

    q_values = valid_results[
        "MWU_q"
    ].to_numpy(
        dtype=float
    )

    q_values_sorted = np.sort(
        q_values
    )

    ranks = np.arange(
        1,
        len(q_values_sorted) + 1,
    )

    significant_indices = np.where(
        q_values_sorted <= ALPHA_FDR
    )[0]

    number_significant = len(
        significant_indices
    )

    # Prevent zero values from breaking the logarithmic y-axis.
    positive_q_values = q_values_sorted[
        q_values_sorted > 0
    ]

    if len(positive_q_values) > 0:
        minimum_positive_q = float(
            positive_q_values.min()
        )

        lower_limit = max(
            minimum_positive_q * 0.5,
            1e-12,
        )
    else:
        lower_limit = 1e-12

    plotted_q_values = np.clip(
        q_values_sorted,
        lower_limit,
        1.0,
    )

    figure, axis = plt.subplots(
        figsize=(14, 7)
    )

    axis.plot(
        ranks,
        plotted_q_values,
        marker="o",
        linestyle="-",
        markersize=6,
        linewidth=2,
        color="0.2",
        label="Features",
    )

    axis.axhline(
        ALPHA_FDR,
        linestyle="--",
        linewidth=2,
        color="0.5",
        label=(
            f"FDR threshold = "
            f"{ALPHA_FDR}"
        ),
    )

    if number_significant > 0:
        boundary_x = (
            significant_indices.max()
            + 1.5
        )

        axis.axvline(
            boundary_x,
            linestyle=":",
            linewidth=2,
            color="0.35",
        )

        axis.text(
            boundary_x,
            ALPHA_FDR * 1.5,
            (
                f"{number_significant} "
                "significant"
            ),
            rotation=90,
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=20,
        )

    axis.set_yscale(
        "log"
    )

    axis.set_ylim(
        lower_limit,
        1.05,
    )

    axis.set_ylabel(
        "FDR q-value (log scale)"
    )

    axis.set_xlabel(
        "Feature rank (sorted by q-value)"
    )

    axis.set_title(
        "Feature Significance – TUM vs LMU"
    )

    axis.grid(
        True,
        which="both",
        axis="y",
        alpha=0.3,
    )

    axis.legend(
        frameon=False
    )

    figure.tight_layout()

    output_path = (
        OUTPUT_ROOT
        / PLOT_FILENAME
    )

    figure.savefig(
        output_path,
        dpi=PLOT_DPI,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return output_path


# ============================================================
# RESULTS REPORTING
# ============================================================

def print_significant_features(
    results_df: pd.DataFrame,
):
    """
    Print features that significantly differ between centers.
    """
    significant_df = results_df.loc[
        results_df["MWU_q"]
        <= ALPHA_FDR
    ].copy()

    print()
    print(
        "=== SIGNIFICANT TUM–LMU DIFFERENCES "
        f"(FDR <= {ALPHA_FDR:.3f}) ==="
    )

    if significant_df.empty:
        print(
            "No features are significant after FDR correction."
        )

        return significant_df

    for _, row in significant_df.iterrows():
        if row["TUM_Median"] > row["LMU_Median"]:
            direction = "higher in TUM"
        elif row["TUM_Median"] < row["LMU_Median"]:
            direction = "higher in LMU"
        else:
            direction = "equal medians"

        print(
            f"- {row['Feature']}: "
            f"q={row['MWU_q']:.3e}, "
            f"p={row['MWU_p']:.3e}, "
            f"TUM median={row['TUM_Median']:.3f}, "
            f"LMU median={row['LMU_Median']:.3f}, "
            f"{direction}"
        )

    return significant_df


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("TUM VERSUS LMU FEATURE SIGNIFICANCE ANALYSIS")
    print("=" * 80)

    tum_df = load_excel_file(
        excel_path=TUM_EXCEL_PATH,
        dataset_name="TUM",
    )

    lmu_df = load_excel_file(
        excel_path=LMU_EXCEL_PATH,
        dataset_name="LMU",
    )

    (
        numeric_features,
        initially_excluded,
    ) = find_shared_numeric_features(
        tum_df=tum_df,
        lmu_df=lmu_df,
    )

    (
        results_df,
        statistically_skipped_df,
    ) = analyze_features(
        tum_df=tum_df,
        lmu_df=lmu_df,
        numeric_features=numeric_features,
    )

    summary_path = (
        OUTPUT_ROOT
        / SUMMARY_FILENAME
    )

    results_df.to_csv(
        summary_path,
        index=False,
    )

    # Combine discovery-stage and analysis-stage skipped features.
    skipped_frames = []

    if initially_excluded:
        discovery_skipped_df = pd.DataFrame(
            initially_excluded
        )

        discovery_skipped_df["TUM_N"] = np.nan
        discovery_skipped_df["LMU_N"] = np.nan

        discovery_skipped_df = discovery_skipped_df[
            [
                "Feature",
                "TUM_N",
                "LMU_N",
                "Reason",
            ]
        ]

        skipped_frames.append(
            discovery_skipped_df
        )

    if not statistically_skipped_df.empty:
        skipped_frames.append(
            statistically_skipped_df
        )

    if skipped_frames:
        skipped_df = pd.concat(
            skipped_frames,
            axis=0,
            ignore_index=True,
            sort=False,
        )

        skipped_path = (
            OUTPUT_ROOT
            / SKIPPED_FILENAME
        )

        skipped_df.to_csv(
            skipped_path,
            index=False,
        )
    else:
        skipped_df = pd.DataFrame()
        skipped_path = None

    significant_df = print_significant_features(
        results_df
    )

    significant_path = (
        OUTPUT_ROOT
        / SIGNIFICANT_FILENAME
    )

    significant_df.to_csv(
        significant_path,
        index=False,
    )

    plot_path = create_significance_plot(
        results_df
    )

    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(
        f"TUM rows:                    "
        f"{len(tum_df)}"
    )

    print(
        f"LMU rows:                    "
        f"{len(lmu_df)}"
    )

    print(
        f"Shared numeric features:     "
        f"{len(numeric_features)}"
    )

    print(
        f"Features statistically tested: "
        f"{len(results_df)}"
    )

    print(
        f"Features skipped:            "
        f"{len(skipped_df)}"
    )

    print(
        f"Significant after FDR:       "
        f"{len(significant_df)}"
    )

    print(
        f"Summary saved to:            "
        f"{summary_path}"
    )

    print(
        f"Significant features saved:  "
        f"{significant_path}"
    )

    if skipped_path is not None:
        print(
            f"Skipped features saved to:   "
            f"{skipped_path}"
        )

    if plot_path is not None:
        print(
            f"Significance plot saved to:  "
            f"{plot_path}"
        )

    print()
    print("DONE ✓")


if __name__ == "__main__":
    main()