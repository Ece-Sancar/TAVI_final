from pathlib import Path
import math
import re
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

TUM_EXCEL_PATH = Path("./tum.xlsx")
LMU_EXCEL_PATH = Path("./lmu.xlsx")

IMAGE_ROOT = Path("/home/ubuntu/final_dataset")

# Output files
ENTIRE_OUTPUT_PATH = Path("./entire.xlsx")
TUM_CLEANED_OUTPUT_PATH = Path("./tum_cleaned.xlsx")
LMU_CLEANED_OUTPUT_PATH = Path("./lmu_cleaned.xlsx")

SPLIT_OUTPUT_ROOT = Path("./dataset_splits")

# Change this manually if automatic detection selects the wrong column.
# Example:
# ID_COLUMN = "ID"
ID_COLUMN = None

LABEL_COLUMN = "LABEL"
SEX_COLUMN = "SEX"

# Around 20% of each LABEL × SEX group is placed in the final test set.
TEST_FRACTION = 0.20

N_FOLDS = 5
RANDOM_SEED = 42

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

# Candidate names used to automatically find the patient/image ID column.
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

    The function does not remove leading zeros from string IDs.
    """
    if pd.isna(value):
        return None

    value_str = str(value).strip()

    if not value_str:
        return None

    # Remove an image extension if an Excel value contains one.
    suffix = Path(value_str).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        value_str = Path(value_str).stem.strip()

    # Excel often converts integer IDs to floating-point values such as 123.0.
    if re.fullmatch(r"-?\d+\.0+", value_str):
        value_str = value_str.split(".")[0]

    return value_str


def find_id_column(df, requested_column=None):
    """
    Find the ID column.

    If ID_COLUMN is explicitly configured, it is used.
    Otherwise, common ID column names are searched.
    """
    if requested_column is not None:
        if requested_column not in df.columns:
            raise ValueError(
                f"Configured ID column '{requested_column}' was not found.\n"
                f"Available columns:\n{list(df.columns)}"
            )
        return requested_column

    # First try exact candidate matches.
    for candidate in ID_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate

    # Then try case-insensitive candidate matches.
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
    Check that LABEL and SEX are available and contain no missing values.
    """
    required_columns = [LABEL_COLUMN, SEX_COLUMN]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{missing_columns}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    for column in required_columns:
        missing_count = int(df[column].isna().sum())

        if missing_count > 0:
            raise ValueError(
                f"{dataset_name} has {missing_count} missing values "
                f"in '{column}'.\n"
                "Fill or remove these values before constructing "
                "stratified splits."
            )


def collect_image_identifiers(image_root):
    """
    Recursively scan the image directory and collect all image filename stems.

    Example:
        /home/ubuntu/final_dataset/123.png -> "123"
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

        normalized_id = normalize_identifier(path.stem)

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
    Keep only rows whose normalized ID exists among the image filenames.
    """
    cleaned_df = df.copy()

    normalized_ids = cleaned_df[id_column].map(normalize_identifier)

    missing_id_mask = normalized_ids.isna()
    missing_image_mask = ~normalized_ids.isin(image_ids)
    remove_mask = missing_id_mask | missing_image_mask

    removed_df = cleaned_df.loc[remove_mask].copy()
    cleaned_df = cleaned_df.loc[~remove_mask].copy()

    cleaned_df.reset_index(drop=True, inplace=True)
    removed_df.reset_index(drop=True, inplace=True)

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


def make_stratification_key(df):
    """
    Construct a LABEL × SEX stratification key.

    Examples:
        label 0, sex F -> "0__F"
        label 1, sex M -> "1__M"
    """
    label_values = df[LABEL_COLUMN].astype(str).str.strip()
    sex_values = df[SEX_COLUMN].astype(str).str.strip()

    return label_values + "__" + sex_values


def allocate_test_count(group_size, test_fraction, n_folds):
    """
    Choose the number of test samples for one LABEL × SEX group.

    The allocation tries to:
      1. approximate TEST_FRACTION;
      2. leave at least one sample per fold;
      3. make the remaining development count divisible by five
         when reasonably possible.

    Making the remaining count divisible by five allows every fold to
    receive exactly the same count from that stratum.
    """
    if group_size < n_folds + 1:
        raise ValueError(
            f"A stratification group contains only {group_size} samples. "
            f"At least {n_folds + 1} samples are needed to create a "
            "non-empty test split and five development folds."
        )

    target_test_count = group_size * test_fraction

    possible_counts = []

    # At least 1 test sample, and at least n_folds development samples.
    for test_count in range(1, group_size - n_folds + 1):
        remaining_count = group_size - test_count

        # Prefer allocations where each fold receives the exact same
        # number from this stratum.
        divisible = remaining_count % n_folds == 0

        distance_from_target = abs(test_count - target_test_count)

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

    The test and five folds are mutually exclusive.
    Their union equals the complete cleaned dataset.
    """
    df = df.copy().reset_index(drop=True)
    df["_STRATIFICATION_KEY"] = make_stratification_key(df)

    rng = np.random.default_rng(random_seed)

    test_indices = []
    fold_indices = [[] for _ in range(N_FOLDS)]

    grouped = df.groupby(
        "_STRATIFICATION_KEY",
        sort=True,
        dropna=False,
    )

    print()
    print(f"Constructing splits for {dataset_name}")
    print("-" * 60)

    for stratum_name, stratum_df in grouped:
        indices = stratum_df.index.to_numpy().copy()
        rng.shuffle(indices)

        group_size = len(indices)

        test_count = allocate_test_count(
            group_size=group_size,
            test_fraction=TEST_FRACTION,
            n_folds=N_FOLDS,
        )

        stratum_test_indices = indices[:test_count]
        stratum_development_indices = indices[test_count:]

        test_indices.extend(stratum_test_indices.tolist())

        # Shuffle development samples one more time before distributing.
        rng.shuffle(stratum_development_indices)

        # array_split guarantees that fold sizes differ by at most one.
        stratum_fold_parts = np.array_split(
            stratum_development_indices,
            N_FOLDS,
        )

        for fold_number, fold_part in enumerate(stratum_fold_parts):
            fold_indices[fold_number].extend(fold_part.tolist())

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

    # Shuffle the order of rows inside every output split.
    test_indices = np.asarray(test_indices)
    rng.shuffle(test_indices)

    shuffled_fold_indices = []

    for indices in fold_indices:
        indices = np.asarray(indices)
        rng.shuffle(indices)
        shuffled_fold_indices.append(indices)

    test_df = (
        df.loc[test_indices]
        .drop(columns="_STRATIFICATION_KEY")
        .reset_index(drop=True)
    )

    folds = []

    for indices in shuffled_fold_indices:
        fold_df = (
            df.loc[indices]
            .drop(columns="_STRATIFICATION_KEY")
            .reset_index(drop=True)
        )
        folds.append(fold_df)

    validate_partition(
        original_df=df.drop(columns="_STRATIFICATION_KEY"),
        test_df=test_df,
        folds=folds,
        dataset_name=dataset_name,
    )

    return test_df, folds


def row_identity_set(df):
    """
    Create a row-level identity representation for validation.

    This verifies that no row is lost or duplicated across partitions.
    """
    normalized = df.copy()

    for column in normalized.columns:
        normalized[column] = normalized[column].map(
            lambda value: "<NA>" if pd.isna(value) else str(value)
        )

    return set(map(tuple, normalized.to_numpy()))


def validate_partition(
    original_df,
    test_df,
    folds,
    dataset_name,
):
    """
    Validate that:
      - test and folds do not overlap;
      - folds do not overlap with each other;
      - every original row appears exactly once;
      - total row counts match.
    """
    split_dataframes = [test_df] + folds
    split_names = ["test"] + [
        f"fold{index}"
        for index in range(1, N_FOLDS + 1)
    ]

    total_split_rows = sum(len(split_df) for split_df in split_dataframes)

    if total_split_rows != len(original_df):
        raise AssertionError(
            f"{dataset_name}: split sizes sum to {total_split_rows}, "
            f"but the original dataset contains {len(original_df)} rows."
        )

    identity_sets = [
        row_identity_set(split_df)
        for split_df in split_dataframes
    ]

    for i in range(len(identity_sets)):
        for j in range(i + 1, len(identity_sets)):
            overlap = identity_sets[i].intersection(identity_sets[j])

            if overlap:
                raise AssertionError(
                    f"{dataset_name}: overlap detected between "
                    f"{split_names[i]} and {split_names[j]}."
                )

    union_set = set().union(*identity_sets)
    original_set = row_identity_set(original_df)

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
    Save:
        test.xlsx
        fold1.xlsx
        ...
        fold5.xlsx
    """
    output_directory.mkdir(parents=True, exist_ok=True)

    test_df.to_excel(
        output_directory / "test.xlsx",
        index=False,
    )

    for fold_number, fold_df in enumerate(folds, start=1):
        fold_df.to_excel(
            output_directory / f"fold{fold_number}.xlsx",
            index=False,
        )


def concatenate_matching_splits(
    tum_test,
    tum_folds,
    lmu_test,
    lmu_folds,
):
    """
    Build merged splits only by concatenating matching TUM and LMU splits.

    Therefore:
        merged test  = TUM test  + LMU test
        merged fold1 = TUM fold1 + LMU fold1
        ...
    """
    merged_test = pd.concat(
        [tum_test, lmu_test],
        axis=0,
        ignore_index=True,
    )

    merged_folds = []

    for tum_fold, lmu_fold in zip(tum_folds, lmu_folds):
        merged_fold = pd.concat(
            [tum_fold, lmu_fold],
            axis=0,
            ignore_index=True,
        )

        merged_folds.append(merged_fold)

    return merged_test, merged_folds


def distribution_table(df):
    """
    Return LABEL × SEX counts for a split.
    """
    table = (
        df.groupby(
            [LABEL_COLUMN, SEX_COLUMN],
            dropna=False,
        )
        .size()
        .reset_index(name="COUNT")
        .sort_values(
            [LABEL_COLUMN, SEX_COLUMN],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    table["PERCENT"] = (
        table["COUNT"] / len(df) * 100
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
    print(f"{dataset_name} SPLIT SUMMARY")
    print("=" * 80)

    all_splits = [("test", test_df)] + [
        (f"fold{index}", fold)
        for index, fold in enumerate(folds, start=1)
    ]

    for split_name, split_df in all_splits:
        print()
        print(f"{split_name}: {len(split_df)} samples")
        print(distribution_table(split_df).to_string(index=False))


def verify_merged_split(
    tum_df,
    lmu_df,
    merged_df,
    split_name,
):
    """
    Verify that a merged split contains exactly the concatenated TUM and
    LMU split rows.
    """
    expected_rows = len(tum_df) + len(lmu_df)

    if len(merged_df) != expected_rows:
        raise AssertionError(
            f"{split_name}: expected {expected_rows} merged rows, "
            f"but found {len(merged_df)}."
        )

    expected_set = row_identity_set(
        pd.concat([tum_df, lmu_df], ignore_index=True)
    )
    merged_set = row_identity_set(merged_df)

    if expected_set != merged_set:
        raise AssertionError(
            f"{split_name}: merged contents do not equal "
            "TUM + LMU contents."
        )

def print_dataset_statistics(df, dataset_name):
    """
    Print:
      - total number of samples;
      - number and percentage for each SEX value;
      - number and percentage for each LABEL value;
      - number and percentage for each LABEL × SEX combination.
    """
    print()
    print("=" * 80)
    print(f"{dataset_name} DATASET STATISTICS")
    print("=" * 80)

    total_samples = len(df)
    print(f"Total samples: {total_samples}")

    print()
    print("SEX distribution")
    print("-" * 40)

    sex_counts = df[SEX_COLUMN].value_counts(
        dropna=False,
        sort=False,
    )

    for sex_value, count in sex_counts.items():
        percentage = 100 * count / total_samples if total_samples else 0

        print(
            f"SEX={sex_value}: "
            f"{count} samples "
            f"({percentage:.2f}%)"
        )

    print()
    print("LABEL distribution")
    print("-" * 40)

    label_counts = df[LABEL_COLUMN].value_counts(
        dropna=False,
        sort=False,
    )

    for label_value, count in label_counts.items():
        percentage = 100 * count / total_samples if total_samples else 0

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
            [LABEL_COLUMN, SEX_COLUMN],
            dropna=False,
        )
        .size()
        .reset_index(name="COUNT")
        .sort_values(
            [LABEL_COLUMN, SEX_COLUMN],
            kind="stable",
        )
    )

    for _, row in joint_counts.iterrows():
        count = int(row["COUNT"])
        percentage = 100 * count / total_samples if total_samples else 0

        print(
            f"LABEL={row[LABEL_COLUMN]}, "
            f"SEX={row[SEX_COLUMN]}: "
            f"{count} samples "
            f"({percentage:.2f}%)"
        )

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("DATASET CLEANING AND FIXED SPLIT GENERATION")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Load Excel files
    # --------------------------------------------------------
    if not TUM_EXCEL_PATH.exists():
        raise FileNotFoundError(
            f"TUM Excel file does not exist: {TUM_EXCEL_PATH}"
        )

    if not LMU_EXCEL_PATH.exists():
        raise FileNotFoundError(
            f"LMU Excel file does not exist: {LMU_EXCEL_PATH}"
        )

    tum_df = pd.read_excel(TUM_EXCEL_PATH)
    lmu_df = pd.read_excel(LMU_EXCEL_PATH)

    validate_required_columns(tum_df, "TUM")
    validate_required_columns(lmu_df, "LMU")

    tum_id_column = find_id_column(tum_df, ID_COLUMN)
    lmu_id_column = find_id_column(lmu_df, ID_COLUMN)

    print(f"TUM ID column: {tum_id_column}")
    print(f"LMU ID column: {lmu_id_column}")

    # The two files should normally have the same schema.
    if list(tum_df.columns) != list(lmu_df.columns):
        print()
        print("WARNING:")
        print("TUM and LMU do not have identical column order/schema.")
        print("Merged data will use the union of their columns.")
        print("Missing columns in either dataset will contain NaN.")

    # --------------------------------------------------------
    # 2. Find all available images
    # --------------------------------------------------------
    image_ids = collect_image_identifiers(IMAGE_ROOT)

    # --------------------------------------------------------
    # 3. Remove rows without images
    # --------------------------------------------------------
    tum_cleaned, tum_removed = clean_dataframe_using_images(
        df=tum_df,
        image_ids=image_ids,
        id_column=tum_id_column,
        dataset_name="TUM",
    )

    lmu_cleaned, lmu_removed = clean_dataframe_using_images(
        df=lmu_df,
        image_ids=image_ids,
        id_column=lmu_id_column,
        dataset_name="LMU",
    )

    # Save removed rows for inspection.
    removed_output_directory = SPLIT_OUTPUT_ROOT / "removed_rows"
    removed_output_directory.mkdir(parents=True, exist_ok=True)

    tum_removed.to_excel(
        removed_output_directory / "tum_removed_missing_images.xlsx",
        index=False,
    )

    lmu_removed.to_excel(
        removed_output_directory / "lmu_removed_missing_images.xlsx",
        index=False,
    )

    # Save cleaned source datasets.
    tum_cleaned.to_excel(
        TUM_CLEANED_OUTPUT_PATH,
        index=False,
    )

    lmu_cleaned.to_excel(
        LMU_CLEANED_OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # 4. Create complete merged dataset
    # --------------------------------------------------------
    entire_df = pd.concat(
        [tum_cleaned, lmu_cleaned],
        axis=0,
        ignore_index=True,
    )

    entire_df.to_excel(
        ENTIRE_OUTPUT_PATH,
        index=False,
    )

    # Print statistics for the complete cleaned datasets.
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
    # 5. Create fixed domain-specific splits
    # --------------------------------------------------------
    tum_test, tum_folds = create_fixed_test_and_folds(
        df=tum_cleaned,
        dataset_name="TUM",
        random_seed=RANDOM_SEED,
    )

    # Use another deterministic seed so that random ordering is independent.
    lmu_test, lmu_folds = create_fixed_test_and_folds(
        df=lmu_cleaned,
        dataset_name="LMU",
        random_seed=RANDOM_SEED + 1,
    )

    # --------------------------------------------------------
    # 6. Create merged splits as exact TUM + LMU combinations
    # --------------------------------------------------------
    merged_test, merged_folds = concatenate_matching_splits(
        tum_test=tum_test,
        tum_folds=tum_folds,
        lmu_test=lmu_test,
        lmu_folds=lmu_folds,
    )

    # --------------------------------------------------------
    # 7. Verify merged identities
    # --------------------------------------------------------
    verify_merged_split(
        tum_df=tum_test,
        lmu_df=lmu_test,
        merged_df=merged_test,
        split_name="merged test",
    )

    for fold_index in range(N_FOLDS):
        verify_merged_split(
            tum_df=tum_folds[fold_index],
            lmu_df=lmu_folds[fold_index],
            merged_df=merged_folds[fold_index],
            split_name=f"merged fold{fold_index + 1}",
        )

    validate_partition(
        original_df=entire_df,
        test_df=merged_test,
        folds=merged_folds,
        dataset_name="MERGED",
    )

    # --------------------------------------------------------
    # 8. Save split files
    # --------------------------------------------------------
    save_split_collection(
        output_directory=SPLIT_OUTPUT_ROOT / "tum",
        test_df=tum_test,
        folds=tum_folds,
    )

    save_split_collection(
        output_directory=SPLIT_OUTPUT_ROOT / "lmu",
        test_df=lmu_test,
        folds=lmu_folds,
    )

    save_split_collection(
        output_directory=SPLIT_OUTPUT_ROOT / "merged",
        test_df=merged_test,
        folds=merged_folds,
    )

    # --------------------------------------------------------
    # 9. Print detailed distributions
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
    # 10. Final concise summary
    # --------------------------------------------------------
    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(f"TUM samples left:       {len(tum_cleaned)}")
    print(f"LMU samples left:       {len(lmu_cleaned)}")
    print(f"Merged samples:         {len(entire_df)}")

    print()
    print(f"TUM test samples:       {len(tum_test)}")
    print(f"LMU test samples:       {len(lmu_test)}")
    print(f"Merged test samples:    {len(merged_test)}")

    print()
    print(f"Cleaned TUM saved to:   {TUM_CLEANED_OUTPUT_PATH}")
    print(f"Cleaned LMU saved to:   {LMU_CLEANED_OUTPUT_PATH}")
    print(f"Entire dataset saved:   {ENTIRE_OUTPUT_PATH}")
    print(f"Split folders saved:    {SPLIT_OUTPUT_ROOT}")

    print()
    print("Merged test and folds were verified as exact combinations:")
    print("  merged/test.xlsx  = tum/test.xlsx  + lmu/test.xlsx")
    print("  merged/fold1.xlsx = tum/fold1.xlsx + lmu/fold1.xlsx")
    print("  ...")
    print("  merged/fold5.xlsx = tum/fold5.xlsx + lmu/fold5.xlsx")


if __name__ == "__main__":
    main()