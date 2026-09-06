from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

TUM_EXCEL_PATH = Path("./tum.xlsx")
LMU_EXCEL_PATH = Path("./lmu.xlsx")

IMAGE_ROOT = Path("/home/ubuntu/final_dataset")

# ECG interval source files.
TUM_ECG_INTERVAL_PATH = Path(
    "/home/ubuntu/TAVI_new/dataset/dataset_new_binary.xlsx"
)

LMU_ECG_INTERVAL_PATH = Path(
    "/home/ubuntu/TAVI_final/dataset/ecg/ecg_intervals_lmu/ecg.xlsx"
)

# Output files.
ENTIRE_OUTPUT_PATH = Path("./entire.xlsx")
TUM_CLEANED_OUTPUT_PATH = Path("./tum_cleaned.xlsx")
LMU_CLEANED_OUTPUT_PATH = Path("./lmu_cleaned.xlsx")

SPLIT_OUTPUT_ROOT = Path("./dataset_splits")

# Distribution plot output paths.
PQ_DISTRIBUTION_PLOT_PATH = Path("./pq_distribution.png")
QRS_DISTRIBUTION_PLOT_PATH = Path("./qrs_distribution.png")

# Change manually if automatic ID-column detection is incorrect.
#
# Example:
# ID_COLUMN = "ID"
ID_COLUMN = None

LABEL_COLUMN = "LABEL"
SEX_COLUMN = "SEX"

QRS_COLUMN = "QRSADM"
PQ_COLUMN = "PQADM"

# Around 20% of each LABEL × SEX group goes into the fixed test set.
TEST_FRACTION = 0.20

N_FOLDS = 5
RANDOM_SEED = 42

# Histogram configuration.
HISTOGRAM_BINS = 50
PLOT_DPI = 300

# Large presentation-friendly fonts.
FONT_SIZE = 20
TICK_FONT_SIZE = 18
LEGEND_FONT_SIZE = 18
LINE_WIDTH = 3.0

# Image file extensions that will be considered.
IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

# Candidate names used to automatically find an ID column.
ID_COLUMN_CANDIDATES = [
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
]

# ============================================================
# DATA-SIZE EXPERIMENT CONFIGURATION
# ============================================================

DATA_SIZE_PERCENTAGES = [
    2,
    5,
    10,
    20,
    50,
]

DATA_SIZE_OUTPUT_ROOT = (
    SPLIT_OUTPUT_ROOT
    / "data_size"
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def normalize_identifier(value):
    """
    Convert an Excel ID or image filename stem into a comparable string.

    Examples:
        123       -> "123"
        123.0     -> "123"
        " 123 "   -> "123"
        "00123"   -> "00123"
        "123.png" -> "123"

    Leading zeros in string IDs are preserved.
    """
    if pd.isna(value):
        return None

    value_str = str(value).strip()

    if not value_str:
        return None

    # Remove image extension if the Excel ID includes one.
    suffix = Path(value_str).suffix.lower()

    if suffix in IMAGE_EXTENSIONS:
        value_str = Path(value_str).stem.strip()

    # Excel often represents integer IDs as values such as 123.0.
    if re.fullmatch(r"-?\d+\.0+", value_str):
        value_str = value_str.split(".")[0]

    return value_str


def find_id_column(df, requested_column=None):
    """
    Find the ID column in a dataframe.
    """
    if requested_column is not None:
        if requested_column not in df.columns:
            raise ValueError(
                f"Configured ID column '{requested_column}' was not found.\n"
                f"Available columns:\n{list(df.columns)}"
            )

        return requested_column

    # Exact candidate matches.
    for candidate in ID_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate

    # Case-insensitive candidate matches.
    lower_to_original = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in ID_COLUMN_CANDIDATES:
        candidate_lower = candidate.lower()

        if candidate_lower in lower_to_original:
            return lower_to_original[candidate_lower]

    raise ValueError(
        "Could not automatically determine the ID column.\n"
        "Set ID_COLUMN at the top of the script.\n"
        f"Available columns:\n{list(df.columns)}"
    )


def validate_required_columns(df, dataset_name):
    """
    Check that LABEL and SEX exist and have no missing values.
    """
    required_columns = [
        LABEL_COLUMN,
        SEX_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{missing_columns}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    for column in required_columns:
        missing_count = int(
            df[column].isna().sum()
        )

        if missing_count > 0:
            raise ValueError(
                f"{dataset_name} has {missing_count} missing values "
                f"in '{column}'.\n"
                "Fill or remove these values before constructing "
                "stratified splits."
            )


def collect_image_identifiers(image_root):
    """
    Recursively scan the image directory and collect image filename stems.
    """
    if not image_root.exists():
        raise FileNotFoundError(
            f"Image directory does not exist: {image_root}"
        )

    image_ids = set()
    image_file_count = 0

    for path in image_root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        normalized_id = normalize_identifier(
            path.stem
        )

        if normalized_id is not None:
            image_ids.add(normalized_id)
            image_file_count += 1

    if image_file_count == 0:
        raise RuntimeError(
            f"No supported images were found under: {image_root}"
        )

    print(
        f"Found {image_file_count} image files representing "
        f"{len(image_ids)} unique image IDs."
    )

    return image_ids


def clean_dataframe_using_images(
    df,
    image_ids,
    id_column,
    dataset_name,
):
    """
    Keep only rows whose normalized ID exists among image filenames.
    """
    cleaned_df = df.copy()

    normalized_ids = cleaned_df[
        id_column
    ].map(normalize_identifier)

    missing_id_mask = normalized_ids.isna()
    missing_image_mask = ~normalized_ids.isin(
        image_ids
    )

    remove_mask = (
        missing_id_mask
        | missing_image_mask
    )

    removed_df = cleaned_df.loc[
        remove_mask
    ].copy()

    cleaned_df = cleaned_df.loc[
        ~remove_mask
    ].copy()

    cleaned_df.reset_index(
        drop=True,
        inplace=True,
    )

    removed_df.reset_index(
        drop=True,
        inplace=True,
    )

    print()
    print(f"{dataset_name} image matching")
    print("-" * 60)
    print(f"Original rows:             {len(df)}")
    print(f"Rows removed:              {len(removed_df)}")
    print(f"Rows remaining:            {len(cleaned_df)}")
    print(
        f"Rows with missing IDs:     "
        f"{int(missing_id_mask.sum())}"
    )

    if len(removed_df) > 0:
        missing_examples = (
            removed_df[id_column]
            .head(20)
            .astype(str)
            .tolist()
        )

        print(
            "First missing/unmatched IDs: "
            + ", ".join(missing_examples)
        )

    return cleaned_df, removed_df


# ============================================================
# ECG INTERVAL MATCHING
# ============================================================

def load_interval_table(
    interval_path,
    dataset_name,
):
    """
    Load an ECG interval file and retain:
        ID
        QRSADM
        PQADM

    Duplicate interval IDs are rejected because they would produce
    ambiguous matches.
    """
    if not interval_path.exists():
        raise FileNotFoundError(
            f"{dataset_name} ECG interval file does not exist: "
            f"{interval_path}"
        )

    interval_df = pd.read_excel(
        interval_path
    )

    interval_id_column = find_id_column(
        interval_df,
        ID_COLUMN,
    )

    missing_columns = [
        column
        for column in [
            QRS_COLUMN,
            PQ_COLUMN,
        ]
        if column not in interval_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} ECG interval file is missing columns: "
            f"{missing_columns}\n"
            f"Available columns:\n{list(interval_df.columns)}"
        )

    interval_df = interval_df[
        [
            interval_id_column,
            QRS_COLUMN,
            PQ_COLUMN,
        ]
    ].copy()

    interval_df["_NORMALIZED_ID"] = (
        interval_df[
            interval_id_column
        ].map(normalize_identifier)
    )

    # Remove interval rows without a usable ID.
    interval_df = interval_df.loc[
        interval_df["_NORMALIZED_ID"].notna()
    ].copy()

    duplicate_mask = interval_df[
        "_NORMALIZED_ID"
    ].duplicated(
        keep=False
    )

    if duplicate_mask.any():
        duplicate_ids = (
            interval_df.loc[
                duplicate_mask,
                "_NORMALIZED_ID",
            ]
            .drop_duplicates()
            .head(20)
            .tolist()
        )

        raise ValueError(
            f"{dataset_name} ECG interval file contains duplicate IDs.\n"
            f"First duplicate IDs: {duplicate_ids}"
        )

    interval_df[QRS_COLUMN] = pd.to_numeric(
        interval_df[QRS_COLUMN],
        errors="coerce",
    )

    interval_df[PQ_COLUMN] = pd.to_numeric(
        interval_df[PQ_COLUMN],
        errors="coerce",
    )

    interval_df = interval_df[
        [
            "_NORMALIZED_ID",
            QRS_COLUMN,
            PQ_COLUMN,
        ]
    ].copy()

    print()
    print(f"{dataset_name} ECG interval source")
    print("-" * 60)
    print(f"Source file:               {interval_path}")
    print(f"Rows with valid ID:        {len(interval_df)}")
    print(
        f"Rows with QRSADM:          "
        f"{int(interval_df[QRS_COLUMN].notna().sum())}"
    )
    print(
        f"Rows with PQADM:           "
        f"{int(interval_df[PQ_COLUMN].notna().sum())}"
    )

    return interval_df


def add_ecg_intervals(
    dataset_df,
    dataset_id_column,
    interval_df,
    dataset_name,
):
    """
    Add QRSADM and PQADM to a dataset by matching normalized IDs.

    This performs a left join, so dataset rows are not removed when an
    interval is unavailable.
    """
    output_df = dataset_df.copy()

    # Remove existing versions to avoid merge suffixes.
    columns_to_drop = [
        column
        for column in [
            QRS_COLUMN,
            PQ_COLUMN,
        ]
        if column in output_df.columns
    ]

    if columns_to_drop:
        print(
            f"{dataset_name}: replacing existing columns "
            f"{columns_to_drop}"
        )

        output_df = output_df.drop(
            columns=columns_to_drop
        )

    output_df["_NORMALIZED_ID"] = (
        output_df[
            dataset_id_column
        ].map(normalize_identifier)
    )

    output_df = output_df.merge(
        interval_df,
        how="left",
        on="_NORMALIZED_ID",
        validate="many_to_one",
    )

    matched_qrs = int(
        output_df[
            QRS_COLUMN
        ].notna().sum()
    )

    matched_pq = int(
        output_df[
            PQ_COLUMN
        ].notna().sum()
    )

    both_matched = int(
        (
            output_df[QRS_COLUMN].notna()
            & output_df[PQ_COLUMN].notna()
        ).sum()
    )

    neither_matched = int(
        (
            output_df[QRS_COLUMN].isna()
            & output_df[PQ_COLUMN].isna()
        ).sum()
    )

    print()
    print(f"{dataset_name} ECG interval matching")
    print("-" * 60)
    print(f"Dataset rows:              {len(output_df)}")
    print(f"QRSADM matched:            {matched_qrs}")
    print(f"PQADM matched:             {matched_pq}")
    print(f"Both values matched:       {both_matched}")
    print(f"No interval values:        {neither_matched}")

    output_df = output_df.drop(
        columns="_NORMALIZED_ID"
    )

    # Store as nullable integers to preserve empty cells in Excel.
    output_df[QRS_COLUMN] = (
        pd.to_numeric(
            output_df[QRS_COLUMN],
            errors="coerce",
        )
        .round()
        .astype("Int64")
    )

    output_df[PQ_COLUMN] = (
        pd.to_numeric(
            output_df[PQ_COLUMN],
            errors="coerce",
        )
        .round()
        .astype("Int64")
    )

    return output_df


# ============================================================
# DISTRIBUTION PLOTS
# ============================================================

def prepare_numeric_values(
    df,
    column,
):
    """
    Return finite numerical values from a dataframe column.
    """
    values = pd.to_numeric(
        df[column],
        errors="coerce",
    ).dropna().to_numpy(
        dtype=float
    )

    return values[
        np.isfinite(values)
    ]


def determine_common_histogram_bins(
    tum_values,
    lmu_values,
    number_of_bins,
):
    """
    Create common histogram bins for TUM and LMU.
    """
    combined_values = np.concatenate(
        [
            tum_values,
            lmu_values,
        ]
    )

    if len(combined_values) == 0:
        raise ValueError(
            "No valid values were available for plotting."
        )

    minimum = float(
        np.min(combined_values)
    )

    maximum = float(
        np.max(combined_values)
    )

    if minimum == maximum:
        minimum -= 0.5
        maximum += 0.5

    return np.linspace(
        minimum,
        maximum,
        number_of_bins + 1,
    )


def plot_domain_distribution(
    tum_df,
    lmu_df,
    column,
    x_label,
    output_path,
):
    """
    Plot TUM and LMU distributions as two grayscale step histograms.

    Density normalization is used because TUM and LMU contain different
    numbers of samples.
    """
    tum_values = prepare_numeric_values(
        tum_df,
        column,
    )

    lmu_values = prepare_numeric_values(
        lmu_df,
        column,
    )

    if len(tum_values) == 0:
        print(
            f"WARNING: no valid TUM values for {column}. "
            f"Plot was not created."
        )
        return

    if len(lmu_values) == 0:
        print(
            f"WARNING: no valid LMU values for {column}. "
            f"Plot was not created."
        )
        return

    bins = determine_common_histogram_bins(
        tum_values=tum_values,
        lmu_values=lmu_values,
        number_of_bins=HISTOGRAM_BINS,
    )

    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "font.family": "sans-serif",
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": TICK_FONT_SIZE,
            "ytick.labelsize": TICK_FONT_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
        }
    )

    figure, axis = plt.subplots(
        figsize=(10, 7)
    )

    axis.hist(
        tum_values,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=LINE_WIDTH,
        color="black",
        label=f"TUM (n={len(tum_values)})",
    )

    axis.hist(
        lmu_values,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=LINE_WIDTH,
        color="0.55",
        linestyle="--",
        label=f"LMU (n={len(lmu_values)})",
    )

    axis.set_xlabel(
        x_label
    )

    axis.set_ylabel(
        "Density"
    )

    # No title, as requested.
    axis.legend(
        frameon=False
    )

    axis.spines["top"].set_visible(
        False
    )

    axis.spines["right"].set_visible(
        False
    )

    axis.grid(
        axis="y",
        linestyle=":",
        linewidth=1.0,
        color="0.80",
    )

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=PLOT_DPI,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        f"{column} distribution plot saved to: "
        f"{output_path}"
    )


# ============================================================
# SPLIT CONSTRUCTION
# ============================================================

def make_stratification_key(df):
    """
    Construct a LABEL × SEX stratification key.
    """
    label_values = (
        df[LABEL_COLUMN]
        .astype(str)
        .str.strip()
    )

    sex_values = (
        df[SEX_COLUMN]
        .astype(str)
        .str.strip()
    )

    return (
        label_values
        + "__"
        + sex_values
    )


def allocate_test_count(
    group_size,
    test_fraction,
    n_folds,
):
    """
    Choose the number of test samples for one LABEL × SEX group.

    The allocation:
      1. approximates TEST_FRACTION;
      2. leaves at least one sample per development fold;
      3. prefers a remaining development count divisible by five.
    """
    if group_size < n_folds + 1:
        raise ValueError(
            f"A stratification group contains only {group_size} samples. "
            f"At least {n_folds + 1} samples are needed to create a "
            "non-empty test split and five development folds."
        )

    target_test_count = (
        group_size
        * test_fraction
    )

    possible_counts = []

    for test_count in range(
        1,
        group_size - n_folds + 1,
    ):
        remaining_count = (
            group_size
            - test_count
        )

        divisible = (
            remaining_count
            % n_folds
            == 0
        )

        distance_from_target = abs(
            test_count
            - target_test_count
        )

        possible_counts.append(
            (
                0 if divisible else 1,
                distance_from_target,
                test_count,
            )
        )

    possible_counts.sort()

    return possible_counts[0][2]


def create_fixed_test_and_folds(
    df,
    dataset_name,
    random_seed,
):
    """
    Split one domain into:
        test
        fold1
        fold2
        fold3
        fold4
        fold5

    Splitting is performed independently inside every LABEL × SEX group.

    Validation is performed using original row indices, so nullable values,
    duplicate row contents, and pandas dtype differences do not cause
    false validation failures.
    """
    df = df.copy().reset_index(drop=True)

    df["_STRATIFICATION_KEY"] = make_stratification_key(
        df
    )

    rng = np.random.default_rng(
        random_seed
    )

    test_indices = []
    fold_indices = [
        []
        for _ in range(N_FOLDS)
    ]

    grouped = df.groupby(
        "_STRATIFICATION_KEY",
        sort=True,
        dropna=False,
    )

    print()
    print(
        f"Constructing splits for {dataset_name}"
    )
    print("-" * 60)

    for stratum_name, stratum_df in grouped:
        indices = (
            stratum_df.index
            .to_numpy(dtype=np.int64)
            .copy()
        )

        rng.shuffle(
            indices
        )

        group_size = len(
            indices
        )

        test_count = allocate_test_count(
            group_size=group_size,
            test_fraction=TEST_FRACTION,
            n_folds=N_FOLDS,
        )

        stratum_test_indices = (
            indices[:test_count]
        )

        stratum_development_indices = (
            indices[test_count:]
        )

        test_indices.extend(
            stratum_test_indices.tolist()
        )

        rng.shuffle(
            stratum_development_indices
        )

        stratum_fold_parts = np.array_split(
            stratum_development_indices,
            N_FOLDS,
        )

        for fold_number, fold_part in enumerate(
            stratum_fold_parts
        ):
            fold_indices[
                fold_number
            ].extend(
                fold_part.tolist()
            )

        fold_sizes = [
            len(fold_part)
            for fold_part in stratum_fold_parts
        ]

        print(
            f"Stratum {stratum_name}: "
            f"total={group_size}, "
            f"test={test_count}, "
            f"folds={fold_sizes}"
        )

    # Convert all index collections to NumPy arrays.
    test_indices = np.asarray(
        test_indices,
        dtype=np.int64,
    )

    fold_indices = [
        np.asarray(
            indices,
            dtype=np.int64,
        )
        for indices in fold_indices
    ]

    # Shuffle final row order inside each output split.
    rng.shuffle(
        test_indices
    )

    for indices in fold_indices:
        rng.shuffle(
            indices
        )

    # --------------------------------------------------------
    # Validate the partition using row indices
    # --------------------------------------------------------
    validate_partition_indices(
        original_row_count=len(df),
        test_indices=test_indices,
        fold_indices=fold_indices,
        dataset_name=dataset_name,
    )

    # --------------------------------------------------------
    # Construct output dataframes
    # --------------------------------------------------------
    test_df = (
        df.iloc[test_indices]
        .drop(
            columns="_STRATIFICATION_KEY"
        )
        .reset_index(drop=True)
    )

    folds = []

    for indices in fold_indices:
        fold_df = (
            df.iloc[indices]
            .drop(
                columns="_STRATIFICATION_KEY"
            )
            .reset_index(drop=True)
        )

        folds.append(
            fold_df
        )

    return test_df, folds

def validate_partition_indices(
    original_row_count,
    test_indices,
    fold_indices,
    dataset_name,
):
    """
    Validate that test and fold index arrays form an exact partition of
    the original dataframe.

    Checks:
      - total number of selected rows is correct;
      - no row index appears more than once;
      - no original row is missing;
      - no invalid row index is present.
    """
    all_index_arrays = [
        np.asarray(
            test_indices,
            dtype=np.int64,
        ),
        *[
            np.asarray(
                indices,
                dtype=np.int64,
            )
            for indices in fold_indices
        ],
    ]

    all_indices = np.concatenate(
        all_index_arrays
    )

    # Total count check.
    if len(all_indices) != original_row_count:
        raise AssertionError(
            f"{dataset_name}: partition contains "
            f"{len(all_indices)} rows, but the original dataset "
            f"contains {original_row_count} rows."
        )

    # Bounds check.
    invalid_indices = all_indices[
        (all_indices < 0)
        | (all_indices >= original_row_count)
    ]

    if len(invalid_indices) > 0:
        raise AssertionError(
            f"{dataset_name}: invalid row indices were found: "
            f"{invalid_indices[:20].tolist()}"
        )

    unique_indices, occurrence_counts = np.unique(
        all_indices,
        return_counts=True,
    )

    duplicated_indices = unique_indices[
        occurrence_counts > 1
    ]

    if len(duplicated_indices) > 0:
        raise AssertionError(
            f"{dataset_name}: {len(duplicated_indices)} rows occur "
            "in more than one split.\n"
            f"First duplicated row indices: "
            f"{duplicated_indices[:20].tolist()}"
        )

    expected_indices = np.arange(
        original_row_count,
        dtype=np.int64,
    )

    missing_indices = np.setdiff1d(
        expected_indices,
        unique_indices,
    )

    if len(missing_indices) > 0:
        raise AssertionError(
            f"{dataset_name}: {len(missing_indices)} rows are missing "
            "from the test/fold partition.\n"
            f"First missing row indices: "
            f"{missing_indices[:20].tolist()}"
        )

    unexpected_indices = np.setdiff1d(
        unique_indices,
        expected_indices,
    )

    if len(unexpected_indices) > 0:
        raise AssertionError(
            f"{dataset_name}: unexpected row indices were found.\n"
            f"First unexpected indices: "
            f"{unexpected_indices[:20].tolist()}"
        )

    print(
        f"{dataset_name}: partition validation passed. "
        f"All {original_row_count} rows occur exactly once."
    )


def row_identity_set(df):
    """
    Create a row-level identity representation for validation.
    """
    normalized = df.copy()

    for column in normalized.columns:
        normalized[column] = (
            normalized[column].map(
                lambda value: (
                    "<NA>"
                    if pd.isna(value)
                    else str(value)
                )
            )
        )

    return set(
        map(
            tuple,
            normalized.to_numpy(),
        )
    )


def validate_partition(
    original_df,
    test_df,
    folds,
    dataset_name,
):
    """
    Validate that test and folds do not overlap and cover all rows.
    """
    split_dataframes = [
        test_df
    ] + folds

    split_names = [
        "test"
    ] + [
        f"fold{index}"
        for index in range(
            1,
            N_FOLDS + 1,
        )
    ]

    total_split_rows = sum(
        len(split_df)
        for split_df in split_dataframes
    )

    if total_split_rows != len(
        original_df
    ):
        raise AssertionError(
            f"{dataset_name}: split sizes sum to {total_split_rows}, "
            f"but the original dataset contains {len(original_df)} rows."
        )

    identity_sets = [
        row_identity_set(
            split_df
        )
        for split_df in split_dataframes
    ]

    for i in range(
        len(identity_sets)
    ):
        for j in range(
            i + 1,
            len(identity_sets),
        ):
            overlap = identity_sets[
                i
            ].intersection(
                identity_sets[j]
            )

            if overlap:
                raise AssertionError(
                    f"{dataset_name}: overlap detected between "
                    f"{split_names[i]} and {split_names[j]}."
                )

    union_set = set().union(
        *identity_sets
    )

    original_set = row_identity_set(
        original_df
    )

    if union_set != original_set:
        raise AssertionError(
            f"{dataset_name}: the union of test and folds does not "
            "match the original cleaned dataset."
        )


def save_split_collection(
    output_directory,
    test_df,
    folds,
):
    """
    Save test.xlsx and fold1.xlsx through fold5.xlsx.
    """
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    test_df.to_excel(
        output_directory
        / "test.xlsx",
        index=False,
    )

    for fold_number, fold_df in enumerate(
        folds,
        start=1,
    ):
        fold_df.to_excel(
            output_directory
            / f"fold{fold_number}.xlsx",
            index=False,
        )


def concatenate_matching_splits(
    tum_test,
    tum_folds,
    lmu_test,
    lmu_folds,
):
    """
    Build merged splits by concatenating corresponding TUM and LMU splits.
    """
    merged_test = pd.concat(
        [
            tum_test,
            lmu_test,
        ],
        axis=0,
        ignore_index=True,
    )

    merged_folds = []

    for tum_fold, lmu_fold in zip(
        tum_folds,
        lmu_folds,
    ):
        merged_fold = pd.concat(
            [
                tum_fold,
                lmu_fold,
            ],
            axis=0,
            ignore_index=True,
        )

        merged_folds.append(
            merged_fold
        )

    return (
        merged_test,
        merged_folds,
    )

# ============================================================
# DATA-SIZE EXPERIMENT DATASETS
# ============================================================

def normalize_binary_label(value):
    """
    Normalize LABEL values only for balanced sampling.

    The original LABEL values in the dataframe are NOT changed.
    """
    value = str(value).strip().lower()

    mapping = {
        "no event": 0,
        "no_event": 0,
        "noevent": 0,
        "0": 0,
        "pacer": 1,
        "pacemaker": 1,
        "1": 1,
    }

    if value not in mapping:
        raise ValueError(
            f"Unsupported LABEL value for data-size sampling: {value!r}"
        )

    return mapping[value]


def attach_original_fold_number(
    folds,
):
    """
    Combine the five development folds while remembering which
    original fold every row belongs to.

    test.xlsx is never included.
    """
    parts = []

    for fold_number, fold_df in enumerate(
        folds,
        start=1,
    ):
        part = fold_df.copy()

        part["_ORIGINAL_FOLD"] = fold_number

        parts.append(part)

    development_df = pd.concat(
        parts,
        axis=0,
        ignore_index=True,
    )

    development_df["_BINARY_LABEL"] = (
        development_df[LABEL_COLUMN]
        .map(normalize_binary_label)
    )

    return development_df


def calculate_balanced_subset_size(
    total_development_samples,
    percentage,
    n_folds,
):
    """
    Determine a usable sample count for a percentage experiment.

    Requirements:
      1. Approximately matches the requested percentage.
      2. Exactly balanced between labels.
      3. Can be divided equally across all folds.
      4. Every fold receives the same number from each label.

    Therefore total sample count must be divisible by:

        2 labels × n_folds

    For 5 folds:
        total must be divisible by 10.
    """

    requested = (
        total_development_samples
        * percentage
        / 100.0
    )

    required_multiple = (
        2 * n_folds
    )

    # Find the nearest valid multiple.
    lower = (
        int(requested)
        // required_multiple
        * required_multiple
    )

    upper = (
        lower
        + required_multiple
    )

    # Need at least one sample of each label in every fold.
    lower = max(
        lower,
        required_multiple,
    )

    upper = max(
        upper,
        required_multiple,
    )

    if abs(lower - requested) <= abs(
        upper - requested
    ):
        selected_total = lower
    else:
        selected_total = upper

    return selected_total   


def sample_balanced_development_subset(
    folds,
    percentage,
    random_seed,
    dataset_name,
):
    """
    Create one balanced data-size experiment dataset.

    The five original development folds are first combined.

    Then:
      1. A percentage-sized balanced subset is selected.
      2. The selected subset is re-shuffled.
      3. It is divided into five NEW equal-sized folds.
      4. Every fold receives the same number of No Event
         and Pacemaker samples.

    The original fold membership is intentionally NOT preserved.

    This is preferable for very small data-size experiments because
    otherwise some folds can be empty or contain only one class.
    """

    # --------------------------------------------------------
    # Combine all five original development folds
    # --------------------------------------------------------
    development_df = pd.concat(
        folds,
        axis=0,
        ignore_index=True,
        sort=False,
    )

    development_df = development_df.copy()

    development_df["_BINARY_LABEL"] = (
        development_df[
            LABEL_COLUMN
        ].map(
            normalize_binary_label
        )
    )

    total_development_samples = len(
        development_df
    )

    # --------------------------------------------------------
    # Determine valid subset size
    # --------------------------------------------------------
    target_total = calculate_balanced_subset_size(
        total_development_samples=(
            total_development_samples
        ),
        percentage=percentage,
        n_folds=N_FOLDS,
    )

    samples_per_label = (
        target_total // 2
    )

    samples_per_label_per_fold = (
        samples_per_label
        // N_FOLDS
    )

    samples_per_fold = (
        target_total
        // N_FOLDS
    )

    # --------------------------------------------------------
    # Separate labels
    # --------------------------------------------------------
    label_0_df = (
        development_df.loc[
            development_df[
                "_BINARY_LABEL"
            ]
            == 0
        ]
        .copy()
    )

    label_1_df = (
        development_df.loc[
            development_df[
                "_BINARY_LABEL"
            ]
            == 1
        ]
        .copy()
    )

    available_per_label = min(
        len(label_0_df),
        len(label_1_df),
    )

    if samples_per_label > available_per_label:
        raise ValueError(
            f"{dataset_name} {percentage}%: "
            f"need {samples_per_label} samples per label, "
            f"but only {available_per_label} are available."
        )

    # --------------------------------------------------------
    # Sample equal numbers from the two labels
    # --------------------------------------------------------
    sampled_label_0 = (
        label_0_df.sample(
            n=samples_per_label,
            replace=False,
            random_state=(
                random_seed
                + percentage * 100
                + 1
            ),
        )
        .reset_index(drop=True)
    )

    sampled_label_1 = (
        label_1_df.sample(
            n=samples_per_label,
            replace=False,
            random_state=(
                random_seed
                + percentage * 100
                + 2
            ),
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Shuffle each class independently
    # --------------------------------------------------------
    sampled_label_0 = (
        sampled_label_0.sample(
            frac=1.0,
            random_state=(
                random_seed
                + percentage * 1000
                + 10
            ),
        )
        .reset_index(drop=True)
    )

    sampled_label_1 = (
        sampled_label_1.sample(
            frac=1.0,
            random_state=(
                random_seed
                + percentage * 1000
                + 20
            ),
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Build five equal folds
    #
    # Every fold receives exactly:
    #
    #   samples_per_label_per_fold No Event
    #   samples_per_label_per_fold Pacemaker
    #
    # --------------------------------------------------------
    percentage_folds = []

    for fold_index in range(
        N_FOLDS
    ):
        start = (
            fold_index
            * samples_per_label_per_fold
        )

        end = (
            start
            + samples_per_label_per_fold
        )

        fold_label_0 = (
            sampled_label_0.iloc[
                start:end
            ]
            .copy()
        )

        fold_label_1 = (
            sampled_label_1.iloc[
                start:end
            ]
            .copy()
        )

        fold_df = pd.concat(
            [
                fold_label_0,
                fold_label_1,
            ],
            axis=0,
            ignore_index=True,
        )

        # Shuffle row order inside the fold.
        fold_df = (
            fold_df.sample(
                frac=1.0,
                random_state=(
                    random_seed
                    + percentage * 10000
                    + fold_index
                ),
            )
            .drop(
                columns=[
                    "_BINARY_LABEL",
                ]
            )
            .reset_index(drop=True)
        )

        percentage_folds.append(
            fold_df
        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------
    fold_sizes = [
        len(fold)
        for fold in percentage_folds
    ]

    if len(
        set(fold_sizes)
    ) != 1:
        raise AssertionError(
            f"{dataset_name} {percentage}%: "
            f"fold sizes are not equal: {fold_sizes}"
        )

    for fold_number, fold_df in enumerate(
        percentage_folds,
        start=1,
    ):
        fold_labels = (
            fold_df[
                LABEL_COLUMN
            ]
            .map(
                normalize_binary_label
            )
        )

        label_counts = (
            fold_labels
            .value_counts()
            .to_dict()
        )

        count_0 = int(
            label_counts.get(
                0,
                0,
            )
        )

        count_1 = int(
            label_counts.get(
                1,
                0,
            )
        )

        if count_0 != count_1:
            raise AssertionError(
                f"{dataset_name} {percentage}% "
                f"fold{fold_number} is not balanced: "
                f"No Event={count_0}, "
                f"Pacemaker={count_1}"
            )

    actual_total = sum(
        fold_sizes
    )

    actual_percentage = (
        actual_total
        / total_development_samples
        * 100.0
    )

    print()
    print("=" * 70)
    print(
        f"{dataset_name} — "
        f"{percentage}% DATA-SIZE DATASET"
    )
    print("=" * 70)

    print(
        f"Original development samples: "
        f"{total_development_samples}"
    )

    print(
        f"Requested percentage:          "
        f"{percentage}%"
    )

    print(
        f"Selected samples:              "
        f"{actual_total}"
    )

    print(
        f"Actual percentage:             "
        f"{actual_percentage:.2f}%"
    )

    print(
        f"No Event samples:              "
        f"{samples_per_label}"
    )

    print(
        f"Pacemaker samples:             "
        f"{samples_per_label}"
    )

    print(
        f"Samples per fold:              "
        f"{samples_per_fold}"
    )

    print(
        f"No Event per fold:             "
        f"{samples_per_label_per_fold}"
    )

    print(
        f"Pacemaker per fold:            "
        f"{samples_per_label_per_fold}"
    )

    print(
        f"Fold sizes:                    "
        f"{fold_sizes}"
    )

    return percentage_folds

def verify_balanced_labels(
    folds,
    dataset_name,
    percentage,
):
    """
    Verify that the union of the five percentage folds contains
    exactly the same number of samples for both labels.
    """
    combined = pd.concat(
        folds,
        axis=0,
        ignore_index=True,
    )

    labels = (
        combined[LABEL_COLUMN]
        .map(normalize_binary_label)
    )

    counts = labels.value_counts().to_dict()

    count_0 = int(
        counts.get(0, 0)
    )

    count_1 = int(
        counts.get(1, 0)
    )

    if count_0 != count_1:
        raise AssertionError(
            f"{dataset_name} {percentage}% is not label-balanced: "
            f"No Event={count_0}, Pacemaker={count_1}"
        )

    print(
        f"{dataset_name} {percentage}% balance verified: "
        f"{count_0} No Event + {count_1} Pacemaker."
    )


def verify_percentage_subset_of_original_folds(
    percentage_folds,
    original_folds,
    dataset_name,
    percentage,
):
    """
    Verify that every sample in each percentage fold comes from
    the corresponding original fold.

    Validation is performed using normalized patient IDs rather than
    whole-row string comparison. This avoids false mismatches caused
    by pandas dtype changes such as 123 vs 123.0.
    """

    for fold_number in range(N_FOLDS):

        percentage_fold = percentage_folds[
            fold_number
        ]

        original_fold = original_folds[
            fold_number
        ]

        # Empty percentage folds are valid for very small percentages.
        if len(percentage_fold) == 0:
            print(
                f"{dataset_name} {percentage}% fold{fold_number + 1}: "
                "empty — valid."
            )
            continue

        percentage_id_column = find_id_column(
            percentage_fold,
            ID_COLUMN,
        )

        original_id_column = find_id_column(
            original_fold,
            ID_COLUMN,
        )

        percentage_ids = set(
            percentage_fold[
                percentage_id_column
            ]
            .map(normalize_identifier)
            .dropna()
            .tolist()
        )

        original_ids = set(
            original_fold[
                original_id_column
            ]
            .map(normalize_identifier)
            .dropna()
            .tolist()
        )

        invalid_ids = sorted(
            percentage_ids
            - original_ids
        )

        if invalid_ids:
            raise AssertionError(
                f"{dataset_name} {percentage}% fold"
                f"{fold_number + 1} contains IDs not present "
                f"in the corresponding original fold.\n"
                f"Invalid IDs: {invalid_ids[:20]}"
            )

        # Also make sure IDs were not somehow duplicated.
        normalized_percentage_ids = (
            percentage_fold[
                percentage_id_column
            ]
            .map(normalize_identifier)
            .dropna()
        )

        duplicate_mask = (
            normalized_percentage_ids
            .duplicated(keep=False)
        )

        if duplicate_mask.any():
            duplicate_ids = (
                normalized_percentage_ids[
                    duplicate_mask
                ]
                .drop_duplicates()
                .tolist()
            )

            raise AssertionError(
                f"{dataset_name} {percentage}% fold"
                f"{fold_number + 1} contains duplicate IDs:\n"
                f"{duplicate_ids[:20]}"
            )

        print(
            f"{dataset_name} {percentage}% fold{fold_number + 1}: "
            f"{len(percentage_fold)} samples verified."
        )

    print(
        f"{dataset_name} {percentage}%: "
        "all selected samples belong to their original folds."
    )

def save_data_size_split_collection(
    output_directory,
    test_df,
    folds,
):
    """
    Save the percentage-specific folds together with the SAME
    original fixed test.xlsx.
    """
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Same fixed test set for every percentage.
    test_df.to_excel(
        output_directory
        / "test.xlsx",
        index=False,
    )

    for fold_number, fold_df in enumerate(
        folds,
        start=1,
    ):
        fold_df.to_excel(
            output_directory
            / f"fold{fold_number}.xlsx",
            index=False,
        )


def create_all_data_size_experiment_datasets(
    tum_test,
    tum_folds,
    lmu_test,
    lmu_folds,
):
    """
    Generate:

        1%
        2%
        5%
        10%
        20%
        50%

    for TUM, LMU, and merged.

    Guarantees:
      - same original test set for every percentage;
      - balanced LABEL distribution for TUM and LMU;
      - merged test = TUM test + LMU test;
      - merged foldX = TUM foldX + LMU foldX.
    """
    print()
    print("=" * 80)
    print("CREATING DATA-SIZE EXPERIMENT DATASETS")
    print("=" * 80)

    for percentage in DATA_SIZE_PERCENTAGES:

        # ----------------------------------------------------
        # TUM subset
        # ----------------------------------------------------
        tum_percentage_folds = (
            sample_balanced_development_subset(
                folds=tum_folds,
                percentage=percentage,
                random_seed=(
                    RANDOM_SEED
                ),
                dataset_name="TUM",
            )
        )

        # ----------------------------------------------------
        # LMU subset
        # ----------------------------------------------------
        lmu_percentage_folds = (
            sample_balanced_development_subset(
                folds=lmu_folds,
                percentage=percentage,
                random_seed=(
                    RANDOM_SEED + 1
                ),
                dataset_name="LMU",
            )
        )

        verify_balanced_labels(
            folds=tum_percentage_folds,
            dataset_name="TUM",
            percentage=percentage,
        )

        verify_balanced_labels(
            folds=lmu_percentage_folds,
            dataset_name="LMU",
            percentage=percentage,
        )

        # ----------------------------------------------------
        # Merged subset
        #
        # IMPORTANT:
        # merged foldX = TUM foldX + LMU foldX
        # ----------------------------------------------------
        merged_percentage_folds = []

        for fold_index in range(
            N_FOLDS
        ):
            merged_fold = pd.concat(
                [
                    tum_percentage_folds[
                        fold_index
                    ],
                    lmu_percentage_folds[
                        fold_index
                    ],
                ],
                axis=0,
                ignore_index=True,
                sort=False,
            )

            merged_percentage_folds.append(
                merged_fold
            )

        # Same fixed merged test set used for every percentage.
        merged_test = pd.concat(
            [
                tum_test,
                lmu_test,
            ],
            axis=0,
            ignore_index=True,
            sort=False,
        )

        # ----------------------------------------------------
        # Verify merged folds
        # ----------------------------------------------------
        for fold_index in range(
            N_FOLDS
        ):
            verify_merged_split(
                tum_df=(
                    tum_percentage_folds[
                        fold_index
                    ]
                ),
                lmu_df=(
                    lmu_percentage_folds[
                        fold_index
                    ]
                ),
                merged_df=(
                    merged_percentage_folds[
                        fold_index
                    ]
                ),
                split_name=(
                    f"{percentage}% merged "
                    f"fold{fold_index + 1}"
                ),
            )

        verify_merged_split(
            tum_df=tum_test,
            lmu_df=lmu_test,
            merged_df=merged_test,
            split_name=(
                f"{percentage}% merged test"
            ),
        )

        # ----------------------------------------------------
        # Output paths
        # ----------------------------------------------------
        percentage_root = (
            DATA_SIZE_OUTPUT_ROOT
            / f"{percentage}_percent"
        )

        # TUM
        save_data_size_split_collection(
            output_directory=(
                percentage_root
                / "tum"
            ),
            test_df=tum_test,
            folds=tum_percentage_folds,
        )

        # LMU
        save_data_size_split_collection(
            output_directory=(
                percentage_root
                / "lmu"
            ),
            test_df=lmu_test,
            folds=lmu_percentage_folds,
        )

        # MERGED
        save_data_size_split_collection(
            output_directory=(
                percentage_root
                / "merged"
            ),
            test_df=merged_test,
            folds=merged_percentage_folds,
        )

        print()
        print(
            f"{percentage}% datasets saved under:"
        )
        print(
            percentage_root
        )


# ============================================================
# REPORTING
# ============================================================

def distribution_table(df):
    """
    Return LABEL × SEX counts for a split.
    """
    table = (
        df.groupby(
            [
                LABEL_COLUMN,
                SEX_COLUMN,
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="COUNT"
        )
        .sort_values(
            [
                LABEL_COLUMN,
                SEX_COLUMN,
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    table["PERCENT"] = (
        table["COUNT"]
        / len(df)
        * 100
    ).round(2)

    return table


def print_split_summary(
    dataset_name,
    test_df,
    folds,
):
    """
    Print split sizes and LABEL × SEX distributions.
    """
    print()
    print("=" * 80)
    print(
        f"{dataset_name} SPLIT SUMMARY"
    )
    print("=" * 80)

    all_splits = [
        (
            "test",
            test_df,
        )
    ] + [
        (
            f"fold{index}",
            fold,
        )
        for index, fold in enumerate(
            folds,
            start=1,
        )
    ]

    for split_name, split_df in all_splits:
        print()
        print(
            f"{split_name}: "
            f"{len(split_df)} samples"
        )

        print(
            distribution_table(
                split_df
            ).to_string(
                index=False
            )
        )


def verify_merged_split(
    tum_df,
    lmu_df,
    merged_df,
    split_name,
):
    """
    Verify that a merged split exactly equals the concatenation of
    its corresponding TUM and LMU splits, including duplicate rows.
    """
    expected_df = pd.concat(
        [
            tum_df,
            lmu_df,
        ],
        axis=0,
        ignore_index=True,
        sort=False,
    )

    if len(merged_df) != len(expected_df):
        raise AssertionError(
            f"{split_name}: expected {len(expected_df)} rows, "
            f"but found {len(merged_df)}."
        )

    # Align column order before comparison.
    expected_df = expected_df.reindex(
        columns=merged_df.columns
    )

    try:
        pd.testing.assert_frame_equal(
            merged_df.reset_index(drop=True),
            expected_df.reset_index(drop=True),
            check_dtype=False,
            check_like=False,
        )

    except AssertionError as error:
        raise AssertionError(
            f"{split_name}: merged dataframe does not exactly equal "
            f"TUM + LMU.\n{error}"
        ) from error

    print(
        f"{split_name}: verified as exact TUM + LMU concatenation."
    )


def print_dataset_statistics(
    df,
    dataset_name,
):
    """
    Print sample, SEX, LABEL, and LABEL × SEX counts.
    """
    print()
    print("=" * 80)
    print(
        f"{dataset_name} DATASET STATISTICS"
    )
    print("=" * 80)

    total_samples = len(
        df
    )

    print(
        f"Total samples: {total_samples}"
    )

    print()
    print("SEX distribution")
    print("-" * 40)

    sex_counts = df[
        SEX_COLUMN
    ].value_counts(
        dropna=False,
        sort=False,
    )

    for sex_value, count in sex_counts.items():
        percentage = (
            100 * count / total_samples
            if total_samples
            else 0
        )

        print(
            f"SEX={sex_value}: "
            f"{count} samples "
            f"({percentage:.2f}%)"
        )

    print()
    print("LABEL distribution")
    print("-" * 40)

    label_counts = df[
        LABEL_COLUMN
    ].value_counts(
        dropna=False,
        sort=False,
    )

    for label_value, count in label_counts.items():
        percentage = (
            100 * count / total_samples
            if total_samples
            else 0
        )

        print(
            f"LABEL={label_value}: "
            f"{count} samples "
            f"({percentage:.2f}%)"
        )

    print()
    print("LABEL × SEX distribution")
    print("-" * 40)

    joint_counts = (
        df.groupby(
            [
                LABEL_COLUMN,
                SEX_COLUMN,
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="COUNT"
        )
        .sort_values(
            [
                LABEL_COLUMN,
                SEX_COLUMN,
            ],
            kind="stable",
        )
    )

    for _, row in joint_counts.iterrows():
        count = int(
            row["COUNT"]
        )

        percentage = (
            100 * count / total_samples
            if total_samples
            else 0
        )

        print(
            f"LABEL={row[LABEL_COLUMN]}, "
            f"SEX={row[SEX_COLUMN]}: "
            f"{count} samples "
            f"({percentage:.2f}%)"
        )

    print()
    print("ECG interval availability")
    print("-" * 40)

    print(
        f"QRSADM available: "
        f"{int(df[QRS_COLUMN].notna().sum())}"
    )

    print(
        f"PQADM available:  "
        f"{int(df[PQ_COLUMN].notna().sum())}"
    )

    print(
        f"Both available:   "
        f"{int(
            (
                df[QRS_COLUMN].notna()
                & df[PQ_COLUMN].notna()
            ).sum()
        )}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("DATASET CLEANING AND FIXED SPLIT GENERATION")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Load primary TUM and LMU datasets
    # --------------------------------------------------------
    if not TUM_EXCEL_PATH.exists():
        raise FileNotFoundError(
            f"TUM Excel file does not exist: "
            f"{TUM_EXCEL_PATH}"
        )

    if not LMU_EXCEL_PATH.exists():
        raise FileNotFoundError(
            f"LMU Excel file does not exist: "
            f"{LMU_EXCEL_PATH}"
        )

    tum_df = pd.read_excel(
        TUM_EXCEL_PATH
    )

    lmu_df = pd.read_excel(
        LMU_EXCEL_PATH
    )

    validate_required_columns(
        tum_df,
        "TUM",
    )

    validate_required_columns(
        lmu_df,
        "LMU",
    )

    tum_id_column = find_id_column(
        tum_df,
        ID_COLUMN,
    )

    lmu_id_column = find_id_column(
        lmu_df,
        ID_COLUMN,
    )

    print(
        f"TUM ID column: {tum_id_column}"
    )

    print(
        f"LMU ID column: {lmu_id_column}"
    )

    # --------------------------------------------------------
    # 2. Find available images
    # --------------------------------------------------------
    image_ids = collect_image_identifiers(
        IMAGE_ROOT
    )

    # --------------------------------------------------------
    # 3. Remove rows without corresponding images
    # --------------------------------------------------------
    tum_cleaned, tum_removed = (
        clean_dataframe_using_images(
            df=tum_df,
            image_ids=image_ids,
            id_column=tum_id_column,
            dataset_name="TUM",
        )
    )

    lmu_cleaned, lmu_removed = (
        clean_dataframe_using_images(
            df=lmu_df,
            image_ids=image_ids,
            id_column=lmu_id_column,
            dataset_name="LMU",
        )
    )

    # --------------------------------------------------------
    # 4. Load and attach ECG interval values
    # --------------------------------------------------------
    tum_interval_df = load_interval_table(
        interval_path=TUM_ECG_INTERVAL_PATH,
        dataset_name="TUM",
    )

    lmu_interval_df = load_interval_table(
        interval_path=LMU_ECG_INTERVAL_PATH,
        dataset_name="LMU",
    )

    tum_cleaned = add_ecg_intervals(
        dataset_df=tum_cleaned,
        dataset_id_column=tum_id_column,
        interval_df=tum_interval_df,
        dataset_name="TUM",
    )

    lmu_cleaned = add_ecg_intervals(
        dataset_df=lmu_cleaned,
        dataset_id_column=lmu_id_column,
        interval_df=lmu_interval_df,
        dataset_name="LMU",
    )

    # --------------------------------------------------------
    # 5. Save removed and cleaned datasets
    # --------------------------------------------------------
    removed_output_directory = (
        SPLIT_OUTPUT_ROOT
        / "removed_rows"
    )

    removed_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    tum_removed.to_excel(
        removed_output_directory
        / "tum_removed_missing_images.xlsx",
        index=False,
    )

    lmu_removed.to_excel(
        removed_output_directory
        / "lmu_removed_missing_images.xlsx",
        index=False,
    )

    tum_cleaned.to_excel(
        TUM_CLEANED_OUTPUT_PATH,
        index=False,
    )

    lmu_cleaned.to_excel(
        LMU_CLEANED_OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # 6. Create and save merged dataset
    # --------------------------------------------------------
    entire_df = pd.concat(
        [
            tum_cleaned,
            lmu_cleaned,
        ],
        axis=0,
        ignore_index=True,
        sort=False,
    )

    entire_df.to_excel(
        ENTIRE_OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # 7. Create ECG interval distribution plots
    # --------------------------------------------------------
    plot_domain_distribution(
        tum_df=tum_cleaned,
        lmu_df=lmu_cleaned,
        column=PQ_COLUMN,
        x_label="PQ interval (ms)",
        output_path=PQ_DISTRIBUTION_PLOT_PATH,
    )

    plot_domain_distribution(
        tum_df=tum_cleaned,
        lmu_df=lmu_cleaned,
        column=QRS_COLUMN,
        x_label="QRS duration (ms)",
        output_path=QRS_DISTRIBUTION_PLOT_PATH,
    )

    # --------------------------------------------------------
    # 8. Print complete dataset statistics
    # --------------------------------------------------------
    print_dataset_statistics(
        df=tum_cleaned,
        dataset_name="TUM CLEANED DATASET",
    )

    print_dataset_statistics(
        df=lmu_cleaned,
        dataset_name="LMU CLEANED DATASET",
    )

    print_dataset_statistics(
        df=entire_df,
        dataset_name="MERGED CLEANED DATASET",
    )

    # --------------------------------------------------------
    # 9. Create fixed domain-specific test sets and folds
    # --------------------------------------------------------
    tum_test, tum_folds = create_fixed_test_and_folds(
        df=tum_cleaned,
        dataset_name="TUM",
        random_seed=RANDOM_SEED,
    )

    lmu_test, lmu_folds = create_fixed_test_and_folds(
        df=lmu_cleaned,
        dataset_name="LMU",
        random_seed=RANDOM_SEED + 1,
    )

    # --------------------------------------------------------
    # 10. Create merged splits as exact TUM + LMU combinations
    # --------------------------------------------------------
    merged_test, merged_folds = concatenate_matching_splits(
        tum_test=tum_test,
        tum_folds=tum_folds,
        lmu_test=lmu_test,
        lmu_folds=lmu_folds,
    )

    # --------------------------------------------------------
    # 11. Verify merged identities
    # --------------------------------------------------------
    verify_merged_split(
        tum_df=tum_test,
        lmu_df=lmu_test,
        merged_df=merged_test,
        split_name="merged test",
    )

    for fold_index in range(
        N_FOLDS
    ):
        verify_merged_split(
            tum_df=tum_folds[
                fold_index
            ],
            lmu_df=lmu_folds[
                fold_index
            ],
            merged_df=merged_folds[
                fold_index
            ],
            split_name=(
                f"merged fold"
                f"{fold_index + 1}"
            ),
        )

    # --------------------------------------------------------
    # 12. Save split files
    # --------------------------------------------------------
    save_split_collection(
        output_directory=(
            SPLIT_OUTPUT_ROOT
            / "tum"
        ),
        test_df=tum_test,
        folds=tum_folds,
    )

    save_split_collection(
        output_directory=(
            SPLIT_OUTPUT_ROOT
            / "lmu"
        ),
        test_df=lmu_test,
        folds=lmu_folds,
    )

    save_split_collection(
        output_directory=(
            SPLIT_OUTPUT_ROOT
            / "merged"
        ),
        test_df=merged_test,
        folds=merged_folds,
    )

    # --------------------------------------------------------
    # 12b. Create data-size experiment datasets
    # --------------------------------------------------------
    create_all_data_size_experiment_datasets(
        tum_test=tum_test,
        tum_folds=tum_folds,
        lmu_test=lmu_test,
        lmu_folds=lmu_folds,
    )

    # --------------------------------------------------------
    # 13. Print detailed split distributions
    # --------------------------------------------------------
    print_split_summary(
        dataset_name="TUM",
        test_df=tum_test,
        folds=tum_folds,
    )

    print_split_summary(
        dataset_name="LMU",
        test_df=lmu_test,
        folds=lmu_folds,
    )

    print_split_summary(
        dataset_name="MERGED",
        test_df=merged_test,
        folds=merged_folds,
    )

    # --------------------------------------------------------
    # 14. Final concise summary
    # --------------------------------------------------------
    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(
        f"TUM samples left:       "
        f"{len(tum_cleaned)}"
    )

    print(
        f"LMU samples left:       "
        f"{len(lmu_cleaned)}"
    )

    print(
        f"Merged samples:         "
        f"{len(entire_df)}"
    )

    print()
    print(
        f"TUM with QRSADM:        "
        f"{int(tum_cleaned[QRS_COLUMN].notna().sum())}"
    )

    print(
        f"TUM with PQADM:         "
        f"{int(tum_cleaned[PQ_COLUMN].notna().sum())}"
    )

    print(
        f"LMU with QRSADM:        "
        f"{int(lmu_cleaned[QRS_COLUMN].notna().sum())}"
    )

    print(
        f"LMU with PQADM:         "
        f"{int(lmu_cleaned[PQ_COLUMN].notna().sum())}"
    )

    print()
    print(
        f"TUM test samples:       "
        f"{len(tum_test)}"
    )

    print(
        f"LMU test samples:       "
        f"{len(lmu_test)}"
    )

    print(
        f"Merged test samples:    "
        f"{len(merged_test)}"
    )

    print()
    print(
        f"Cleaned TUM saved to:   "
        f"{TUM_CLEANED_OUTPUT_PATH}"
    )

    print(
        f"Cleaned LMU saved to:   "
        f"{LMU_CLEANED_OUTPUT_PATH}"
    )

    print(
        f"Entire dataset saved:   "
        f"{ENTIRE_OUTPUT_PATH}"
    )

    print(
        f"Split folders saved:    "
        f"{SPLIT_OUTPUT_ROOT}"
    )

    print(
        f"PQ plot saved to:       "
        f"{PQ_DISTRIBUTION_PLOT_PATH}"
    )

    print(
        f"QRS plot saved to:      "
        f"{QRS_DISTRIBUTION_PLOT_PATH}"
    )

    print()
    print(
        "Merged test and folds were verified as exact combinations:"
    )

    print(
        "  merged/test.xlsx  = "
        "tum/test.xlsx  + lmu/test.xlsx"
    )

    print(
        "  merged/fold1.xlsx = "
        "tum/fold1.xlsx + lmu/fold1.xlsx"
    )

    print("  ...")

    print(
        "  merged/fold5.xlsx = "
        "tum/fold5.xlsx + lmu/fold5.xlsx"
    )


if __name__ == "__main__":
    main()