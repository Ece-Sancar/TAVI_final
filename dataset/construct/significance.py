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

EXCEL_FILE = Path(
    "/home/ubuntu/TAVI_final/dataset/construct/entire.xlsx"
)

LABEL_COLUMN = "LABEL"

OUTPUT_ROOT = Path(
    "feature_significance_entire"
)

ALPHA_FDR = 0.05

EXCLUDE_COLUMNS = {
    LABEL_COLUMN,
    "ID",
    "PATIENT_ID",
}

# Minimum number of non-missing samples required in each group.
MIN_SAMPLES_PER_GROUP = 2

# Plot settings.
PLOT_DPI = 300

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

# Suppress harmless warnings from statistical libraries.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
)


# ============================================================
# LABEL HANDLING
# ============================================================

def normalize_label(value):
    """
    Convert common numeric and text label representations into:

        No Event
        Pacemaker

    Supported examples:

        0
        0.0
        "0"
        "no event"
        "No Event"

        1
        1.0
        "1"
        "pacer"
        "pacemaker"
    """
    if pd.isna(value):
        return None

    if isinstance(
        value,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    ):
        numeric_value = float(value)

        if numeric_value == 0:
            return "No Event"

        if numeric_value == 1:
            return "Pacemaker"

    text = str(value).strip().lower()

    no_event_values = {
        "0",
        "0.0",
        "no event",
        "no_event",
        "noevent",
        "none",
        "no-event",
    }

    pacemaker_values = {
        "1",
        "1.0",
        "pacer",
        "pacemaker",
        "pace maker",
        "pace-maker",
    }

    if text in no_event_values:
        return "No Event"

    if text in pacemaker_values:
        return "Pacemaker"

    return None


# ============================================================
# STATISTICAL FUNCTIONS
# ============================================================

def calculate_mann_whitney(
    no_event_values,
    pacemaker_values,
):
    """
    Run a two-sided Mann-Whitney U test.

    Returns:
        statistic
        p-value
    """
    no_event_values = pd.to_numeric(
        no_event_values,
        errors="coerce",
    ).dropna()

    pacemaker_values = pd.to_numeric(
        pacemaker_values,
        errors="coerce",
    ).dropna()

    if (
        len(no_event_values) < MIN_SAMPLES_PER_GROUP
        or len(pacemaker_values) < MIN_SAMPLES_PER_GROUP
    ):
        return np.nan, np.nan

    try:
        statistic, p_value = mannwhitneyu(
            no_event_values,
            pacemaker_values,
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
    Apply Benjamini-Hochberg FDR correction only to finite p-values.

    NaN p-values remain NaN.
    """
    p = pd.to_numeric(
        p_values,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    q = np.full(
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

    valid_p_values = p[
        valid_mask
    ]

    number_of_tests = len(
        valid_p_values
    )

    if number_of_tests == 0:
        return pd.Series(
            q,
            index=p_values.index,
        )

    order = np.argsort(
        valid_p_values
    )

    sorted_p_values = valid_p_values[
        order
    ]

    ranks = np.arange(
        1,
        number_of_tests + 1,
        dtype=float,
    )

    sorted_q_values = (
        sorted_p_values
        * number_of_tests
        / ranks
    )

    # Enforce monotonicity.
    sorted_q_values = np.minimum.accumulate(
        sorted_q_values[::-1]
    )[::-1]

    sorted_q_values = np.clip(
        sorted_q_values,
        0.0,
        1.0,
    )

    valid_q_values = np.empty_like(
        sorted_q_values
    )

    valid_q_values[
        order
    ] = sorted_q_values

    q[
        valid_indices
    ] = valid_q_values

    return pd.Series(
        q,
        index=p_values.index,
    )


# ============================================================
# DATA LOADING AND VALIDATION
# ============================================================

def load_and_prepare_data():
    """
    Load the Excel file, normalize labels, and validate both groups.
    """
    if not EXCEL_FILE.exists():
        raise FileNotFoundError(
            f"Excel file does not exist: {EXCEL_FILE}"
        )

    dataframe = pd.read_excel(
        EXCEL_FILE
    )

    if LABEL_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Label column '{LABEL_COLUMN}' was not found.\n"
            f"Available columns:\n{list(dataframe.columns)}"
        )

    print("=" * 80)
    print("FEATURE SIGNIFICANCE ANALYSIS")
    print("=" * 80)
    print(f"Input file: {EXCEL_FILE}")
    print(f"Original rows: {len(dataframe)}")

    print()
    print("Original LABEL values")
    print("-" * 60)

    print(
        dataframe[
            LABEL_COLUMN
        ].value_counts(
            dropna=False
        ).to_string()
    )

    dataframe[
        "_NORMALIZED_LABEL"
    ] = dataframe[
        LABEL_COLUMN
    ].map(
        normalize_label
    )

    unknown_label_count = int(
        dataframe[
            "_NORMALIZED_LABEL"
        ].isna().sum()
    )

    if unknown_label_count > 0:
        unknown_values = (
            dataframe.loc[
                dataframe[
                    "_NORMALIZED_LABEL"
                ].isna(),
                LABEL_COLUMN,
            ]
            .drop_duplicates()
            .head(20)
            .tolist()
        )

        print()
        print(
            f"WARNING: {unknown_label_count} rows had unknown "
            "or missing labels and will be excluded."
        )

        print(
            f"Unknown label examples: {unknown_values}"
        )

    dataframe = dataframe.loc[
        dataframe[
            "_NORMALIZED_LABEL"
        ].isin(
            [
                "No Event",
                "Pacemaker",
            ]
        )
    ].copy()

    dataframe[
        LABEL_COLUMN
    ] = dataframe[
        "_NORMALIZED_LABEL"
    ]

    dataframe.drop(
        columns="_NORMALIZED_LABEL",
        inplace=True,
    )

    if dataframe.empty:
        raise RuntimeError(
            "No rows remain after label normalization."
        )

    group_counts = dataframe[
        LABEL_COLUMN
    ].value_counts()

    if "No Event" not in group_counts:
        raise RuntimeError(
            "No 'No Event' samples were found after label normalization."
        )

    if "Pacemaker" not in group_counts:
        raise RuntimeError(
            "No 'Pacemaker' samples were found after label normalization."
        )

    print()
    print("Normalized LABEL counts")
    print("-" * 60)

    print(
        group_counts.to_string()
    )

    return dataframe


def find_numeric_columns(
    dataframe,
):
    """
    Find numerical feature columns, excluding label and ID columns.
    """
    numeric_columns = []

    for column in dataframe.columns:
        if column in EXCLUDE_COLUMNS:
            continue

        if pd.api.types.is_numeric_dtype(
            dataframe[column]
        ):
            numeric_columns.append(
                column
            )

    if not numeric_columns:
        raise RuntimeError(
            "No numeric feature columns were found."
        )

    print()
    print(
        f"Numeric features discovered: "
        f"{len(numeric_columns)}"
    )

    return numeric_columns


# ============================================================
# FEATURE ANALYSIS
# ============================================================

def analyze_features(
    dataframe,
    numeric_columns,
):
    """
    Run Mann-Whitney U tests for all usable numeric features.
    """
    records = []
    skipped_records = []

    no_event_mask = (
        dataframe[
            LABEL_COLUMN
        ] == "No Event"
    )

    pacemaker_mask = (
        dataframe[
            LABEL_COLUMN
        ] == "Pacemaker"
    )

    for feature in numeric_columns:
        no_event_values = pd.to_numeric(
            dataframe.loc[
                no_event_mask,
                feature,
            ],
            errors="coerce",
        ).dropna()

        pacemaker_values = pd.to_numeric(
            dataframe.loc[
                pacemaker_mask,
                feature,
            ],
            errors="coerce",
        ).dropna()

        no_event_count = len(
            no_event_values
        )

        pacemaker_count = len(
            pacemaker_values
        )

        if (
            no_event_count < MIN_SAMPLES_PER_GROUP
            or pacemaker_count < MIN_SAMPLES_PER_GROUP
        ):
            skipped_records.append(
                {
                    "Feature": feature,
                    "NE_N": no_event_count,
                    "PA_N": pacemaker_count,
                    "Reason": (
                        "Insufficient non-missing values"
                    ),
                }
            )

            print(
                f"Skipping {feature}: "
                f"No Event n={no_event_count}, "
                f"Pacemaker n={pacemaker_count}"
            )

            continue

        statistic, p_value = calculate_mann_whitney(
            no_event_values,
            pacemaker_values,
        )

        if not np.isfinite(
            p_value
        ):
            skipped_records.append(
                {
                    "Feature": feature,
                    "NE_N": no_event_count,
                    "PA_N": pacemaker_count,
                    "Reason": (
                        "Statistical test returned NaN"
                    ),
                }
            )

            print(
                f"Skipping {feature}: "
                "Mann-Whitney test returned NaN"
            )

            continue

        records.append(
            {
                "Feature": feature,
                "NE_N": no_event_count,
                "PA_N": pacemaker_count,
                "MWU_stat": statistic,
                "MWU_p": p_value,
                "NE_Median": float(
                    no_event_values.median()
                ),
                "PA_Median": float(
                    pacemaker_values.median()
                ),
                "NE_Mean": float(
                    no_event_values.mean()
                ),
                "PA_Mean": float(
                    pacemaker_values.mean()
                ),
            }
        )

    if not records:
        raise RuntimeError(
            "No numerical feature had enough valid values in both "
            "groups. Check label encoding and missing values."
        )

    results_dataframe = pd.DataFrame(
        records
    )

    results_dataframe[
        "MWU_q"
    ] = benjamini_hochberg_fdr(
        results_dataframe[
            "MWU_p"
        ]
    )

    results_dataframe = (
        results_dataframe
        .sort_values(
            by=[
                "MWU_q",
                "MWU_p",
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    skipped_dataframe = pd.DataFrame(
        skipped_records
    )

    return (
        results_dataframe,
        skipped_dataframe,
    )


# ============================================================
# PLOTTING
# ============================================================

def create_significance_plot(
    results_dataframe,
):
    """
    Plot FDR-adjusted q-values sorted from smallest to largest.
    """
    valid_results = results_dataframe.loc[
        np.isfinite(
            results_dataframe[
                "MWU_q"
            ]
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

    figure, axis = plt.subplots(
        figsize=(14, 7)
    )

    axis.plot(
        ranks,
        q_values_sorted,
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

    # A logarithmic axis cannot display exact zero.
    positive_q_values = q_values_sorted[
        q_values_sorted > 0
    ]

    if len(positive_q_values) > 0:
        minimum_positive = float(
            positive_q_values.min()
        )

        lower_limit = max(
            minimum_positive * 0.5,
            1e-12,
        )
    else:
        lower_limit = 1e-12

    plotted_q_values = np.clip(
        q_values_sorted,
        lower_limit,
        1.0,
    )

    # Update the line with clipped values for safe log plotting.
    axis.lines[0].set_ydata(
        plotted_q_values
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
        "Feature Significance - Merged Dataset"
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

    plot_path = (
        OUTPUT_ROOT
        / "significance_plot_with_binary.png"
    )

    figure.savefig(
        plot_path,
        dpi=PLOT_DPI,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return plot_path


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = load_and_prepare_data()

    numeric_columns = find_numeric_columns(
        dataframe
    )

    (
        results_dataframe,
        skipped_dataframe,
    ) = analyze_features(
        dataframe=dataframe,
        numeric_columns=numeric_columns,
    )

    summary_path = (
        OUTPUT_ROOT
        / "summary_all_features_new.csv"
    )

    results_dataframe.to_csv(
        summary_path,
        index=False,
    )

    if not skipped_dataframe.empty:
        skipped_path = (
            OUTPUT_ROOT
            / "skipped_features.csv"
        )

        skipped_dataframe.to_csv(
            skipped_path,
            index=False,
        )

        print()
        print(
            f"Skipped feature details saved to: "
            f"{skipped_path}"
        )

    significant = results_dataframe.loc[
        results_dataframe[
            "MWU_q"
        ] <= ALPHA_FDR
    ].copy()

    print()
    print(
        "=== SIGNIFICANT FEATURES "
        f"(FDR <= {ALPHA_FDR:.3f}) ==="
    )

    if significant.empty:
        print(
            "No features are significant after FDR correction."
        )

    else:
        for _, row in significant.iterrows():
            print(
                f"- {row['Feature']} "
                f"(q = {row['MWU_q']:.3e}, "
                f"p = {row['MWU_p']:.3e}, "
                f"No Event median = "
                f"{row['NE_Median']:.3f}, "
                f"Pacemaker median = "
                f"{row['PA_Median']:.3f})"
            )

    plot_path = create_significance_plot(
        results_dataframe
    )

    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(
        f"Rows analyzed:             "
        f"{len(dataframe)}"
    )

    print(
        f"Features discovered:       "
        f"{len(numeric_columns)}"
    )

    print(
        f"Features statistically tested: "
        f"{len(results_dataframe)}"
    )

    print(
        f"Features skipped:          "
        f"{len(skipped_dataframe)}"
    )

    print(
        f"Significant after FDR:     "
        f"{len(significant)}"
    )

    print(
        f"Summary saved to:          "
        f"{summary_path}"
    )

    if plot_path is not None:
        print(
            f"Significance plot saved to: "
            f"{plot_path}"
        )

    print()
    print("DONE ✓")


if __name__ == "__main__":
    main()