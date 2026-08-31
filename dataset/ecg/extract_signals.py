from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import re

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_ROOT = Path("./ecg_crops_lmu")
OUTPUT_ROOT = Path("./ecg_signals_lmu")

ECG_CROP_FOLDER_NAME = "ecg_crops"

SUPPORTED_IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
}

# Process patient folders in parallel.
# Start with half of the available CPU cores.
NUM_WORKERS = 20

print(
    f"Using {NUM_WORKERS} parallel workers "
    f"for {INPUT_ROOT} -> {OUTPUT_ROOT}"
)

# Avoid OpenCV opening multiple threads inside each worker.
cv2.setNumThreads(1)

LEADS_PER_PAGE = 6

# ECG signal area within each row.
SIGNAL_LEFT_FRACTION = 0.075
SIGNAL_RIGHT_FRACTION = 0.995

ROW_TOP_MARGIN_FRACTION = 0.08
ROW_BOTTOM_MARGIN_FRACTION = 0.08

# Waveform and grid thresholds.
DARK_PIXEL_THRESHOLD = 175

GRID_MIN_SATURATION = 8
GRID_MIN_VALUE = 130

MAX_TRACKING_JUMP_FRACTION = 0.35
TRACKING_SEARCH_RADIUS_FRACTION = 0.42

MIN_DARK_COMPONENT_AREA = 3

# ECG paper calibration.
PAPER_SPEED_MM_PER_SECOND = 50.0
GAIN_MM_PER_MV = 10.0

TARGET_SAMPLE_RATE_HZ = 500

# Fixed page-to-lead mapping.
#
# Make sure this ordering matches your LMU PDFs.
PAGE_LEAD_NAMES = {
    1: [
        "I",
        "II",
        "III",
        "aVR",
        "aVL",
        "aVF",
    ],
    2: [
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    ],
}

CANONICAL_12_LEAD_ORDER = [
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

OUTPUT_FILENAME = "ecg_12lead.npy"


# ============================================================
# FILE UTILITIES
# ============================================================

def extract_page_number(path: Path):
    """
    Extract page number from a filename such as:

        page_01_ecg_red-grid.png
        page_02_ecg_red-grid.png
    """
    match = re.search(
        r"page[_-]?(\d+)",
        path.stem,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    return int(match.group(1))


# ============================================================
# RED ECG GRID DETECTION
# ============================================================

def create_red_grid_mask(
    image: np.ndarray,
) -> np.ndarray:
    """
    Detect pale red/pink ECG grid pixels.
    """
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    hue, saturation, value = cv2.split(hsv)

    red_hue = (
        (hue <= 30)
        | (hue >= 150)
    )

    sufficiently_colored = (
        saturation >= GRID_MIN_SATURATION
    )

    sufficiently_bright = (
        value >= GRID_MIN_VALUE
    )

    mask = (
        red_hue
        & sufficiently_colored
        & sufficiently_bright
    ).astype(np.uint8) * 255

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones(
            (2, 2),
            dtype=np.uint8,
        ),
    )

    return mask


def group_consecutive_positions(
    positions: np.ndarray,
) -> np.ndarray:
    """
    Group consecutive positions and return the center of each group.
    """
    if len(positions) == 0:
        return np.array(
            [],
            dtype=np.float32,
        )

    groups = []
    current_group = [int(positions[0])]

    for position in positions[1:]:
        position = int(position)

        if position <= current_group[-1] + 1:
            current_group.append(position)
        else:
            groups.append(current_group)
            current_group = [position]

    groups.append(current_group)

    return np.asarray(
        [
            float(np.mean(group))
            for group in groups
        ],
        dtype=np.float32,
    )


def estimate_small_grid_spacing(
    grid_mask: np.ndarray,
):
    """
    Estimate the distance between adjacent 1-mm grid lines in pixels.
    """
    binary = (
        grid_mask > 0
    ).astype(np.float32)

    row_density = binary.mean(axis=1)
    column_density = binary.mean(axis=0)

    row_threshold = max(
        0.01,
        float(
            np.percentile(
                row_density,
                85,
            )
        ),
    )

    column_threshold = max(
        0.01,
        float(
            np.percentile(
                column_density,
                85,
            )
        ),
    )

    row_positions = np.where(
        row_density >= row_threshold
    )[0]

    column_positions = np.where(
        column_density >= column_threshold
    )[0]

    row_centers = group_consecutive_positions(
        row_positions
    )

    column_centers = group_consecutive_positions(
        column_positions
    )

    distances = []

    if len(row_centers) >= 3:
        distances.extend(
            np.diff(row_centers).tolist()
        )

    if len(column_centers) >= 3:
        distances.extend(
            np.diff(column_centers).tolist()
        )

    if not distances:
        return None

    distances = np.asarray(
        distances,
        dtype=np.float32,
    )

    distances = distances[
        (distances >= 2.0)
        & (distances <= 100.0)
    ]

    if len(distances) == 0:
        return None

    lower_limit = np.percentile(
        distances,
        60,
    )

    smaller_distances = distances[
        distances <= lower_limit
    ]

    if len(smaller_distances) == 0:
        smaller_distances = distances

    return float(
        np.median(smaller_distances)
    )


# ============================================================
# WAVEFORM MASK
# ============================================================

def create_dark_waveform_mask(
    image: np.ndarray,
    grid_mask: np.ndarray,
) -> np.ndarray:
    """
    Detect dark waveform pixels while excluding red-grid pixels.
    """
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    _, saturation, _ = cv2.split(hsv)

    dark_mask = (
        gray <= DARK_PIXEL_THRESHOLD
    )

    low_saturation = (
        saturation <= 150
    )

    not_grid = (
        grid_mask == 0
    )

    waveform_mask = (
        dark_mask
        & low_saturation
        & not_grid
    ).astype(np.uint8) * 255

    (
        number_of_labels,
        labels,
        stats,
        _,
    ) = cv2.connectedComponentsWithStats(
        waveform_mask,
        connectivity=8,
    )

    cleaned = np.zeros_like(
        waveform_mask
    )

    for label_index in range(
        1,
        number_of_labels,
    ):
        component_area = stats[
            label_index,
            cv2.CC_STAT_AREA,
        ]

        if component_area >= MIN_DARK_COMPONENT_AREA:
            cleaned[
                labels == label_index
            ] = 255

    return cleaned


# ============================================================
# LEAD ROWS
# ============================================================

def get_equal_row_bounds(
    image_height: int,
    number_of_rows: int,
):
    """
    Divide a page crop into equal horizontal lead rows.
    """
    boundaries = np.linspace(
        0,
        image_height,
        number_of_rows + 1,
    ).round().astype(int)

    return [
        (
            int(boundaries[index]),
            int(boundaries[index + 1]),
        )
        for index in range(number_of_rows)
    ]


# ============================================================
# WAVEFORM TRACKING
# ============================================================

def contiguous_groups(
    values: np.ndarray,
):
    """
    Split sorted integer positions into contiguous groups.
    """
    if len(values) == 0:
        return []

    groups = []
    current = [int(values[0])]

    for value in values[1:]:
        value = int(value)

        if value <= current[-1] + 1:
            current.append(value)
        else:
            groups.append(
                np.asarray(
                    current,
                    dtype=np.int32,
                )
            )
            current = [value]

    groups.append(
        np.asarray(
            current,
            dtype=np.int32,
        )
    )

    return groups


def select_waveform_position(
    candidate_y_positions: np.ndarray,
    previous_y: float,
    gray_column: np.ndarray,
):
    """
    Select the waveform component closest to the previous trace position.
    """
    groups = contiguous_groups(
        candidate_y_positions
    )

    if not groups:
        return None

    best_position = None
    best_score = None

    for group in groups:
        intensities = (
            255.0
            - gray_column[group].astype(np.float32)
        )

        intensities = np.maximum(
            intensities,
            1.0,
        )

        weighted_position = float(
            np.average(
                group.astype(np.float32),
                weights=intensities,
            )
        )

        distance = abs(
            weighted_position
            - previous_y
        )

        darkness_bonus = (
            float(np.mean(intensities))
            / 255.0
        )

        score = (
            distance
            - 4.0 * darkness_bonus
        )

        if (
            best_score is None
            or score < best_score
        ):
            best_score = score
            best_position = weighted_position

    return best_position


def interpolate_missing_values(
    values: np.ndarray,
):
    """
    Linearly interpolate missing signal positions.
    """
    values = values.astype(
        np.float32
    )

    valid = np.isfinite(values)

    if valid.sum() < 2:
        return None

    x = np.arange(
        len(values),
        dtype=np.float32,
    )

    return np.interp(
        x,
        x[valid],
        values[valid],
    ).astype(np.float32)


def median_smooth(
    signal: np.ndarray,
    kernel_size: int = 5,
):
    """
    Apply a small median filter.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    filtered = cv2.medianBlur(
        signal.astype(
            np.float32
        ).reshape(-1, 1),
        kernel_size,
    )

    return filtered.reshape(-1).astype(
        np.float32
    )


def extract_trace_from_row(
    row_image: np.ndarray,
    row_waveform_mask: np.ndarray,
):
    """
    Follow one ECG waveform from left to right.
    """
    row_height, row_width = (
        row_waveform_mask.shape
    )

    gray = cv2.cvtColor(
        row_image,
        cv2.COLOR_BGR2GRAY,
    )

    left_x = int(
        row_width
        * SIGNAL_LEFT_FRACTION
    )

    right_x = int(
        row_width
        * SIGNAL_RIGHT_FRACTION
    )

    top_y = int(
        row_height
        * ROW_TOP_MARGIN_FRACTION
    )

    bottom_y = int(
        row_height
        * (
            1.0
            - ROW_BOTTOM_MARGIN_FRACTION
        )
    )

    center_y = (
        top_y + bottom_y
    ) / 2.0

    search_radius = max(
        8,
        int(
            row_height
            * TRACKING_SEARCH_RADIUS_FRACTION
        ),
    )

    maximum_jump = max(
        5,
        int(
            row_height
            * MAX_TRACKING_JUMP_FRACTION
        ),
    )

    trace = np.full(
        row_width,
        np.nan,
        dtype=np.float32,
    )

    previous_y = center_y

    for x in range(
        left_x,
        right_x,
    ):
        search_start = max(
            top_y,
            int(
                round(previous_y)
                - search_radius
            ),
        )

        search_end = min(
            bottom_y,
            int(
                round(previous_y)
                + search_radius
                + 1
            ),
        )

        candidate_positions = np.where(
            row_waveform_mask[
                search_start:search_end,
                x,
            ] > 0
        )[0]

        if len(candidate_positions) == 0:
            continue

        candidate_positions = (
            candidate_positions
            + search_start
        )

        selected_y = select_waveform_position(
            candidate_y_positions=(
                candidate_positions
            ),
            previous_y=previous_y,
            gray_column=gray[:, x],
        )

        if selected_y is None:
            continue

        if (
            abs(selected_y - previous_y)
            > maximum_jump
        ):
            continue

        trace[x] = selected_y
        previous_y = selected_y

    working_trace = trace[
        left_x:right_x
    ]

    interpolated = interpolate_missing_values(
        working_trace
    )

    if interpolated is None:
        return None

    return median_smooth(
        interpolated,
        kernel_size=5,
    )


# ============================================================
# CALIBRATION AND RESAMPLING
# ============================================================

def pixel_trace_to_mv(
    trace_y: np.ndarray,
    pixels_per_mm: float,
):
    """
    Convert y-pixel positions into approximate mV values.
    """
    baseline_y = float(
        np.median(trace_y)
    )

    displacement_pixels = (
        baseline_y - trace_y
    )

    displacement_mm = (
        displacement_pixels
        / pixels_per_mm
    )

    signal_mv = (
        displacement_mm
        / GAIN_MM_PER_MV
    )

    return signal_mv.astype(
        np.float32
    )


def resample_signal(
    signal: np.ndarray,
    original_sample_rate_hz: float,
    target_sample_rate_hz: float,
):
    """
    Resample a signal using linear interpolation.
    """
    if len(signal) < 2:
        return None

    duration_seconds = (
        (len(signal) - 1)
        / original_sample_rate_hz
    )

    output_length = max(
        2,
        int(
            round(
                duration_seconds
                * target_sample_rate_hz
            )
        ) + 1,
    )

    old_time = np.linspace(
        0.0,
        duration_seconds,
        len(signal),
        dtype=np.float32,
    )

    new_time = np.linspace(
        0.0,
        duration_seconds,
        output_length,
        dtype=np.float32,
    )

    return np.interp(
        new_time,
        old_time,
        signal,
    ).astype(np.float32)


def equalize_signal_lengths(
    signals,
):
    """
    Trim all available signals to a common length.
    """
    valid_lengths = [
        len(signal)
        for signal in signals
        if signal is not None
    ]

    if not valid_lengths:
        return None

    common_length = min(
        valid_lengths
    )

    result = []

    for signal in signals:
        if signal is None:
            result.append(
                np.full(
                    common_length,
                    np.nan,
                    dtype=np.float32,
                )
            )
        else:
            result.append(
                signal[
                    :common_length
                ].astype(np.float32)
            )

    return np.stack(
        result,
        axis=0,
    )


# ============================================================
# PAGE EXTRACTION
# ============================================================

def process_ecg_crop(
    image_path: Path,
):
    """
    Extract six lead signals from one red-grid ECG crop.
    """
    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    page_number = extract_page_number(
        image_path
    )

    if page_number not in PAGE_LEAD_NAMES:
        raise ValueError(
            f"Unsupported or unknown page number: "
            f"{image_path.name}"
        )

    image_height = image.shape[0]

    grid_mask = create_red_grid_mask(
        image
    )

    pixels_per_mm = estimate_small_grid_spacing(
        grid_mask
    )

    if (
        pixels_per_mm is None
        or pixels_per_mm <= 0
    ):
        raise RuntimeError(
            f"Could not estimate grid spacing: "
            f"{image_path}"
        )

    waveform_mask = create_dark_waveform_mask(
        image=image,
        grid_mask=grid_mask,
    )

    native_sample_rate_hz = (
        pixels_per_mm
        * PAPER_SPEED_MM_PER_SECOND
    )

    row_bounds = get_equal_row_bounds(
        image_height=image_height,
        number_of_rows=LEADS_PER_PAGE,
    )

    lead_names = PAGE_LEAD_NAMES[
        page_number
    ]

    extracted_signals = []

    for row_start, row_end in row_bounds:
        row_image = image[
            row_start:row_end,
            :,
        ]

        row_mask = waveform_mask[
            row_start:row_end,
            :,
        ]

        trace_y = extract_trace_from_row(
            row_image=row_image,
            row_waveform_mask=row_mask,
        )

        if trace_y is None:
            extracted_signals.append(
                None
            )
            continue

        signal_mv = pixel_trace_to_mv(
            trace_y=trace_y,
            pixels_per_mm=pixels_per_mm,
        )

        resampled_signal = resample_signal(
            signal=signal_mv,
            original_sample_rate_hz=(
                native_sample_rate_hz
            ),
            target_sample_rate_hz=(
                TARGET_SAMPLE_RATE_HZ
            ),
        )

        extracted_signals.append(
            resampled_signal
        )

    signal_matrix = equalize_signal_lengths(
        extracted_signals
    )

    if signal_matrix is None:
        raise RuntimeError(
            f"No usable signals extracted from: "
            f"{image_path}"
        )

    return {
        "page_number": page_number,
        "lead_names": lead_names,
        "signals": signal_matrix,
    }


# ============================================================
# PATIENT MERGING
# ============================================================

def merge_patient_pages(
    patient_results,
):
    """
    Merge page-level signals into canonical 12-lead order.
    """
    lead_to_signal = {}

    for page_result in patient_results:
        signals = page_result["signals"]
        lead_names = page_result["lead_names"]

        for lead_index, lead_name in enumerate(
            lead_names
        ):
            if lead_name not in lead_to_signal:
                lead_to_signal[lead_name] = (
                    signals[lead_index]
                )

    if not lead_to_signal:
        return None

    minimum_length = min(
        len(signal)
        for signal in lead_to_signal.values()
    )

    merged = np.full(
        (
            len(CANONICAL_12_LEAD_ORDER),
            minimum_length,
        ),
        np.nan,
        dtype=np.float32,
    )

    for lead_index, lead_name in enumerate(
        CANONICAL_12_LEAD_ORDER
    ):
        if lead_name not in lead_to_signal:
            continue

        merged[lead_index] = (
            lead_to_signal[lead_name][
                :minimum_length
            ]
        )

    return merged


# ============================================================
# DATASET DISCOVERY
# ============================================================

def find_patient_crop_groups():
    """
    Find only files created by the red-grid crop detector.
    """
    crop_directories = sorted(
        path
        for path in INPUT_ROOT.rglob(
            ECG_CROP_FOLDER_NAME
        )
        if path.is_dir()
    )

    patient_groups = []

    for crop_directory in crop_directories:
        image_paths = sorted(
            path
            for path in crop_directory.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_IMAGE_SUFFIXES
                and "red-grid"
                in path.stem.lower()
            )
        )

        if not image_paths:
            continue

        patient_directory = (
            crop_directory.parent
        )

        relative_patient_directory = (
            patient_directory.relative_to(
                INPUT_ROOT
            )
        )

        patient_groups.append(
            (
                relative_patient_directory,
                image_paths,
            )
        )

    return patient_groups


# ============================================================
# PARALLEL WORKER
# ============================================================

def process_patient_group(task):
    """
    Process one patient/PDF folder inside a worker process.
    """
    (
        relative_patient_directory_string,
        image_path_strings,
    ) = task

    relative_patient_directory = Path(
        relative_patient_directory_string
    )

    image_paths = [
        Path(path)
        for path in image_path_strings
    ]

    patient_results = []
    errors = []

    for image_path in image_paths:
        try:
            result = process_ecg_crop(
                image_path
            )

            patient_results.append(
                result
            )

        except Exception as error:
            errors.append(
                f"{image_path.name}: {error}"
            )

    if not patient_results:
        return {
            "patient": str(
                relative_patient_directory
            ),
            "success": False,
            "error": "; ".join(errors),
        }

    merged_signals = merge_patient_pages(
        patient_results
    )

    if merged_signals is None:
        return {
            "patient": str(
                relative_patient_directory
            ),
            "success": False,
            "error": "Could not merge extracted pages.",
        }

    patient_output_directory = (
        OUTPUT_ROOT
        / relative_patient_directory
    )

    patient_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        patient_output_directory
        / OUTPUT_FILENAME
    )

    np.save(
        output_path,
        merged_signals.astype(
            np.float32
        ),
        allow_pickle=False,
    )

    return {
        "patient": str(
            relative_patient_directory
        ),
        "success": True,
        "output_path": str(output_path),
        "shape": tuple(
            merged_signals.shape
        ),
        "error": "; ".join(errors),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(
            f"Input root does not exist: "
            f"{INPUT_ROOT}"
        )

    patient_groups = find_patient_crop_groups()

    if not patient_groups:
        raise RuntimeError(
            "No red-grid ECG crop images were found "
            f"inside {INPUT_ROOT}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    tasks = [
        (
            str(relative_patient_directory),
            [
                str(image_path)
                for image_path in image_paths
            ],
        )
        for (
            relative_patient_directory,
            image_paths,
        ) in patient_groups
    ]

    print("=" * 80)
    print("PARALLEL ECG SIGNAL EXTRACTION")
    print("=" * 80)
    print(f"Input root:       {INPUT_ROOT}")
    print(f"Output root:      {OUTPUT_ROOT}")
    print(f"Patient folders:  {len(tasks)}")
    print(f"Workers:          {NUM_WORKERS}")
    print(f"Output file:      {OUTPUT_FILENAME}")
    print(f"Lead order:       {CANONICAL_12_LEAD_ORDER}")
    print(f"Sample rate:      {TARGET_SAMPLE_RATE_HZ} Hz")
    print(
        f"Paper calibration: "
        f"{PAPER_SPEED_MM_PER_SECOND} mm/s, "
        f"{GAIN_MM_PER_MV} mm/mV"
    )

    successful_files = 0
    failed_patients = []

    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS
    ) as executor:
        future_to_patient = {
            executor.submit(
                process_patient_group,
                task,
            ): task[0]
            for task in tasks
        }

        completed = 0

        for future in as_completed(
            future_to_patient
        ):
            completed += 1

            patient_name = future_to_patient[
                future
            ]

            try:
                result = future.result()

                if result["success"]:
                    successful_files += 1

                    print(
                        f"[{completed}/{len(tasks)}] "
                        f"{result['patient']} | "
                        f"saved | "
                        f"shape={result['shape']}"
                    )

                    if result.get("error"):
                        print(
                            f"  Page warning: "
                            f"{result['error']}"
                        )

                else:
                    failed_patients.append(
                        {
                            "patient": result[
                                "patient"
                            ],
                            "error": result[
                                "error"
                            ],
                        }
                    )

                    print(
                        f"[{completed}/{len(tasks)}] "
                        f"{result['patient']} | FAILED | "
                        f"{result['error']}"
                    )

            except Exception as error:
                failed_patients.append(
                    {
                        "patient": patient_name,
                        "error": (
                            "Worker process failed: "
                            f"{error}"
                        ),
                    }
                )

                print(
                    f"[{completed}/{len(tasks)}] "
                    f"{patient_name} | WORKER ERROR | "
                    f"{error}"
                )

    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(
        f"Patient folders discovered: "
        f"{len(tasks)}"
    )
    print(
        f"NPY files constructed:      "
        f"{successful_files}"
    )
    print(
        f"Failed patients:             "
        f"{len(failed_patients)}"
    )
    print(
        f"Outputs saved under:         "
        f"{OUTPUT_ROOT}"
    )

    if failed_patients:
        failure_path = (
            OUTPUT_ROOT
            / "failed_extractions.txt"
        )

        with failure_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            for item in failed_patients:
                file.write(
                    f"Patient: {item['patient']}\n"
                )
                file.write(
                    f"Error: {item['error']}\n\n"
                )

        print(
            f"Failure details:             "
            f"{failure_path}"
        )


if __name__ == "__main__":
    main()