from pathlib import Path
import re
import shutil

import cv2
import fitz  # PyMuPDF
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PDF_ROOT = Path("./ecg_pdfs_lmu")
OUTPUT_ROOT = Path("./ecg_crops_lmu")

# High resolution helps preserve narrow ECG waveforms.
RENDER_DPI = 400

# Maximum width used only during structural detection.
# The final crop is always taken from the full-resolution image.
DETECTION_MAX_WIDTH = 2000

# Extra space around a structural-detector crop.
CROP_PADDING_FRACTION = 0.015

# Prefer direct detection of the pink/red ECG grid.
USE_RED_GRID_DETECTION = True

# Padding around the detected ECG grid.
GRID_PADDING_X_FRACTION = 0.005
GRID_PADDING_Y_FRACTION = 0.005

# Minimum accepted grid dimensions relative to the whole page.
MIN_GRID_WIDTH_FRACTION = 0.50
MIN_GRID_HEIGHT_FRACTION = 0.30

# Reject structural regions smaller than this fraction of the page.
MIN_REGION_AREA_FRACTION = 0.12

# Reject structural regions larger than this fraction of the page.
# This prevents the whole page from being selected.
MAX_REGION_AREA_FRACTION = 0.88

# Most printed ECG regions are relatively wide.
MIN_REGION_ASPECT_RATIO = 1.15

# If no convincing ECG region is found, optionally save a conservative
# fallback crop after removing the header and page margins.
SAVE_FALLBACK_CROP = True

# Fallback fractions measured relative to the oriented page.
FALLBACK_TOP = 0.12
FALLBACK_BOTTOM = 0.97
FALLBACK_LEFT = 0.02
FALLBACK_RIGHT = 0.98

# Save the full rendered page.
SAVE_RENDERED_PAGES = True

# Save an image with the proposed crop rectangle drawn on the page.
SAVE_DEBUG_IMAGES = True

# Save the detected red-grid mask for debugging.
SAVE_GRID_MASKS = True

# Use Tesseract orientation detection when installed.
USE_TESSERACT_ORIENTATION = True

SUPPORTED_PDF_SUFFIXES = {".pdf"}


# ============================================================
# BASIC UTILITIES
# ============================================================

def safe_name(text: str) -> str:
    """Convert a filename into a filesystem-safe output name."""
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("_")


def rotate_image_clockwise(
    image: np.ndarray,
    degrees: int,
) -> np.ndarray:
    """Rotate an image clockwise by 0, 90, 180, or 270 degrees."""
    degrees %= 360

    if degrees == 0:
        return image

    if degrees == 90:
        return cv2.rotate(
            image,
            cv2.ROTATE_90_CLOCKWISE,
        )

    if degrees == 180:
        return cv2.rotate(
            image,
            cv2.ROTATE_180,
        )

    if degrees == 270:
        return cv2.rotate(
            image,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        )

    raise ValueError(
        f"Unsupported rotation: {degrees}"
    )


def resize_for_detection(image: np.ndarray):
    """
    Resize an image for faster structural detection.

    Returns:
        resized_image
        scale_x from detection image to original image
        scale_y from detection image to original image
    """
    height, width = image.shape[:2]

    if width <= DETECTION_MAX_WIDTH:
        return image.copy(), 1.0, 1.0

    scale = DETECTION_MAX_WIDTH / width

    resized = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )

    resized_height, resized_width = resized.shape[:2]

    return (
        resized,
        width / resized_width,
        height / resized_height,
    )


def longest_true_region(values: np.ndarray):
    """
    Return the start and end indices of the longest consecutive True region.

    Returns:
        (start, end), where end is inclusive,
        or None if no True values exist.
    """
    values = np.asarray(
        values,
        dtype=bool,
    )

    if not np.any(values):
        return None

    padded = np.pad(
        values.astype(np.int8),
        (1, 1),
        mode="constant",
        constant_values=0,
    )

    transitions = np.diff(padded)

    starts = np.where(
        transitions == 1
    )[0]

    ends = (
        np.where(transitions == -1)[0]
        - 1
    )

    lengths = ends - starts + 1
    best_index = int(np.argmax(lengths))

    return (
        int(starts[best_index]),
        int(ends[best_index]),
    )


def smooth_projection(
    values: np.ndarray,
    window_size: int,
):
    """Smooth a one-dimensional projection with a moving average."""
    window_size = max(
        1,
        int(window_size),
    )

    kernel = (
        np.ones(
            window_size,
            dtype=np.float32,
        )
        / window_size
    )

    return np.convolve(
        values.astype(np.float32),
        kernel,
        mode="same",
    )


# ============================================================
# PDF RENDERING
# ============================================================

def render_pdf_page(
    page: fitz.Page,
    dpi: int,
) -> np.ndarray:
    """
    Render one PDF page into a full-resolution BGR OpenCV image.

    PyMuPDF normally applies the PDF page's stored rotation metadata.
    """
    zoom = dpi / 72.0
    matrix = fitz.Matrix(
        zoom,
        zoom,
    )

    pixmap = page.get_pixmap(
        matrix=matrix,
        alpha=False,
    )

    image = np.frombuffer(
        pixmap.samples,
        dtype=np.uint8,
    ).reshape(
        pixmap.height,
        pixmap.width,
        pixmap.n,
    )

    if pixmap.n == 4:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGBA2BGR,
        )
    else:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR,
        )

    return image


# ============================================================
# PAGE ORIENTATION
# ============================================================

def tesseract_orientation(image: np.ndarray):
    """
    Use Tesseract only to estimate page orientation.

    Returns the clockwise correction angle, or None when unavailable.
    This does not save OCR text.
    """
    if not USE_TESSERACT_ORIENTATION:
        return None

    if shutil.which("tesseract") is None:
        return None

    try:
        import pytesseract

        preview, _, _ = resize_for_detection(
            image
        )

        rgb = cv2.cvtColor(
            preview,
            cv2.COLOR_BGR2RGB,
        )

        osd = pytesseract.image_to_osd(
            rgb,
            config="--psm 0",
        )

        match = re.search(
            r"Rotate:\s*(\d+)",
            osd,
        )

        if not match:
            return None

        return int(match.group(1)) % 360

    except Exception:
        return None


def ecg_structure_score(
    image: np.ndarray,
) -> float:
    """
    Estimate how ECG-like a page orientation is.

    ECG sheets tend to contain:
      - many horizontal structures;
      - many vertical structures;
      - a large waveform or grid region;
      - a landscape-like arrangement.
    """
    preview, _, _ = resize_for_detection(
        image
    )

    gray = cv2.cvtColor(
        preview,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        11,
    )

    height, width = binary.shape

    horizontal_kernel_width = max(
        20,
        width // 80,
    )

    vertical_kernel_height = max(
        20,
        height // 80,
    )

    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                horizontal_kernel_width,
                1,
            ),
        ),
    )

    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                1,
                vertical_kernel_height,
            ),
        ),
    )

    horizontal_density = (
        np.count_nonzero(horizontal)
        / horizontal.size
    )

    vertical_density = (
        np.count_nonzero(vertical)
        / vertical.size
    )

    landscape_bonus = (
        1.0
        if width >= height
        else 0.5
    )

    return (
        0.45 * horizontal_density
        + 0.45 * vertical_density
        + 0.10 * landscape_bonus
    )


def orient_page(image: np.ndarray):
    """
    Correct page orientation.

    Strategy:
      1. Use Tesseract orientation when available.
      2. Otherwise, if portrait, compare 90° clockwise and counterclockwise.
      3. If already landscape, keep it as rendered.

    Returns:
        oriented image
        applied clockwise rotation
        method string
    """
    detected_rotation = tesseract_orientation(
        image
    )

    if detected_rotation is not None:
        oriented = rotate_image_clockwise(
            image,
            detected_rotation,
        )

        return (
            oriented,
            detected_rotation,
            "tesseract",
        )

    height, width = image.shape[:2]

    if height > width:
        clockwise = rotate_image_clockwise(
            image,
            90,
        )

        counterclockwise = rotate_image_clockwise(
            image,
            270,
        )

        clockwise_score = ecg_structure_score(
            clockwise
        )

        counterclockwise_score = ecg_structure_score(
            counterclockwise
        )

        if clockwise_score >= counterclockwise_score:
            return (
                clockwise,
                90,
                "structure-score",
            )

        return (
            counterclockwise,
            270,
            "structure-score",
        )

    return (
        image,
        0,
        "page-shape",
    )


# ============================================================
# RED / PINK ECG GRID DETECTION
# ============================================================

def create_pink_red_grid_mask(
    image: np.ndarray,
) -> np.ndarray:
    """
    Detect the pale pink or red grid of printed ECG paper.

    The threshold is intentionally permissive because ECG grids are often
    pale and have low saturation.
    """
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    hue, saturation, value = cv2.split(
        hsv
    )

    # Red appears at both ends of OpenCV's hue range.
    red_hue = (
        (hue <= 25)
        | (hue >= 155)
    )

    # Permit pale ECG-grid colors.
    sufficiently_colored = (
        saturation >= 10
    )

    # Exclude dark waveform and text pixels.
    sufficiently_bright = (
        value >= 135
    )

    grid_mask = (
        red_hue
        & sufficiently_colored
        & sufficiently_bright
    ).astype(np.uint8) * 255

    height, width = grid_mask.shape

    # Ignore the page boundary so that a colored border, debug rectangle,
    # or scanner border cannot be interpreted as the ECG grid.
    border_x = max(
        2,
        int(width * 0.01),
    )

    border_y = max(
        2,
        int(height * 0.01),
    )

    grid_mask[:border_y, :] = 0
    grid_mask[-border_y:, :] = 0
    grid_mask[:, :border_x] = 0
    grid_mask[:, -border_x:] = 0

    # Remove tiny isolated colored artifacts.
    grid_mask = cv2.morphologyEx(
        grid_mask,
        cv2.MORPH_OPEN,
        np.ones(
            (2, 2),
            dtype=np.uint8,
        ),
    )

    return grid_mask


def detect_red_grid_box(
    image: np.ndarray,
):
    """
    Detect the ECG plotting area using the pink/red grid.

    Returns:
        box: (x1, y1, x2, y2), or None
        score: approximate confidence
        mask: detected grid mask
    """
    mask = create_pink_red_grid_mask(
        image
    )

    height, width = mask.shape

    binary_mask = mask > 0

    row_density = binary_mask.mean(
        axis=1
    )

    column_density = binary_mask.mean(
        axis=0
    )

    smoothed_rows = smooth_projection(
        row_density,
        window_size=max(
            15,
            height // 100,
        ),
    )

    smoothed_columns = smooth_projection(
        column_density,
        window_size=max(
            15,
            width // 120,
        ),
    )

    row_threshold = max(
        0.0015,
        float(smoothed_rows.max()) * 0.08,
    )

    column_threshold = max(
        0.0015,
        float(smoothed_columns.max()) * 0.08,
    )

    active_rows = (
        smoothed_rows
        > row_threshold
    )

    active_columns = (
        smoothed_columns
        > column_threshold
    )

    row_region = longest_true_region(
        active_rows
    )

    column_region = longest_true_region(
        active_columns
    )

    if (
        row_region is None
        or column_region is None
    ):
        return None, None, mask

    y1, y2 = row_region
    x1, x2 = column_region

    detected_width = (
        x2 - x1 + 1
    )

    detected_height = (
        y2 - y1 + 1
    )

    width_fraction = (
        detected_width / width
    )

    height_fraction = (
        detected_height / height
    )

    if (
        width_fraction
        < MIN_GRID_WIDTH_FRACTION
    ):
        return None, None, mask

    if (
        height_fraction
        < MIN_GRID_HEIGHT_FRACTION
    ):
        return None, None, mask

    padding_x = int(
        width
        * GRID_PADDING_X_FRACTION
    )

    padding_y = int(
        height
        * GRID_PADDING_Y_FRACTION
    )

    x1 = max(
        0,
        x1 - padding_x,
    )

    y1 = max(
        0,
        y1 - padding_y,
    )

    x2 = min(
        width,
        x2 + padding_x + 1,
    )

    y2 = min(
        height,
        y2 + padding_y + 1,
    )

    region = binary_mask[
        y1:y2,
        x1:x2,
    ]

    grid_density = (
        float(region.mean())
        if region.size
        else 0.0
    )

    score = (
        0.40
        * min(
            width_fraction / 0.90,
            1.0,
        )
        + 0.40
        * min(
            height_fraction / 0.65,
            1.0,
        )
        + 0.20
        * min(
            grid_density / 0.10,
            1.0,
        )
    )

    return (
        (x1, y1, x2, y2),
        score,
        mask,
    )


# ============================================================
# STRUCTURAL ECG REGION DETECTION
# ============================================================

def create_red_grid_mask(
    image: np.ndarray,
) -> np.ndarray:
    """
    Detect stronger pink or red ECG grid pixels.

    This mask is used as part of the structural fallback detector.
    """
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    lower_red_1 = np.array(
        [0, 18, 90]
    )

    upper_red_1 = np.array(
        [18, 255, 255]
    )

    lower_red_2 = np.array(
        [165, 18, 90]
    )

    upper_red_2 = np.array(
        [179, 255, 255]
    )

    mask_1 = cv2.inRange(
        hsv,
        lower_red_1,
        upper_red_1,
    )

    mask_2 = cv2.inRange(
        hsv,
        lower_red_2,
        upper_red_2,
    )

    red_mask = cv2.bitwise_or(
        mask_1,
        mask_2,
    )

    red_mask = cv2.morphologyEx(
        red_mask,
        cv2.MORPH_CLOSE,
        np.ones(
            (5, 5),
            dtype=np.uint8,
        ),
    )

    return red_mask


def create_structure_masks(
    image: np.ndarray,
):
    """
    Create masks describing page structure.

    Returns:
        binary ink mask
        horizontal-line mask
        vertical-line mask
        red-grid mask
    """
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        11,
    )

    height, width = binary.shape

    horizontal_kernel_width = max(
        18,
        width // 100,
    )

    vertical_kernel_height = max(
        18,
        height // 100,
    )

    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                horizontal_kernel_width,
                1,
            ),
        ),
    )

    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                1,
                vertical_kernel_height,
            ),
        ),
    )

    red_grid = create_red_grid_mask(
        image
    )

    return (
        binary,
        horizontal,
        vertical,
        red_grid,
    )


def candidate_region_mask(
    image: np.ndarray,
):
    """
    Produce a coarse binary mask for probable ECG regions.

    This version uses less aggressive dilation than the original script,
    so header and footer text are less likely to be merged with the ECG.
    """
    (
        binary,
        horizontal,
        vertical,
        red_grid,
    ) = create_structure_masks(
        image
    )

    height, width = binary.shape

    line_structure = cv2.bitwise_or(
        horizontal,
        vertical,
    )

    line_structure = cv2.bitwise_or(
        line_structure,
        red_grid,
    )

    expanded_structure = cv2.dilate(
        line_structure,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                max(
                    11,
                    width // 150,
                ),
                max(
                    11,
                    height // 150,
                ),
            ),
        ),
        iterations=1,
    )

    waveform_near_structure = cv2.bitwise_and(
        binary,
        expanded_structure,
    )

    combined = cv2.bitwise_or(
        line_structure,
        waveform_near_structure,
    )

    # Smaller dilation than the original version.
    combined = cv2.dilate(
        combined,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                max(
                    15,
                    width // 70,
                ),
                max(
                    9,
                    height // 100,
                ),
            ),
        ),
        iterations=1,
    )

    # Smaller closing kernel than the original version.
    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                max(
                    21,
                    width // 60,
                ),
                max(
                    11,
                    height // 90,
                ),
            ),
        ),
        iterations=1,
    )

    return (
        combined,
        binary,
        horizontal,
        vertical,
        red_grid,
    )


def score_candidate_box(
    box,
    image_shape,
    binary,
    horizontal,
    vertical,
    red_grid,
):
    """
    Score a proposed structural ECG-region rectangle.
    """
    x, y, width, height = box

    page_height, page_width = (
        image_shape[:2]
    )

    page_area = (
        page_height
        * page_width
    )

    box_area = width * height

    if box_area <= 0:
        return -1.0

    area_fraction = (
        box_area / page_area
    )

    aspect_ratio = (
        width
        / max(height, 1)
    )

    region_binary = binary[
        y:y + height,
        x:x + width,
    ]

    region_horizontal = horizontal[
        y:y + height,
        x:x + width,
    ]

    region_vertical = vertical[
        y:y + height,
        x:x + width,
    ]

    region_red = red_grid[
        y:y + height,
        x:x + width,
    ]

    ink_density = (
        np.count_nonzero(region_binary)
        / max(region_binary.size, 1)
    )

    horizontal_density = (
        np.count_nonzero(region_horizontal)
        / max(region_horizontal.size, 1)
    )

    vertical_density = (
        np.count_nonzero(region_vertical)
        / max(region_vertical.size, 1)
    )

    red_density = (
        np.count_nonzero(region_red)
        / max(region_red.size, 1)
    )

    center_y = (
        y + height / 2
    ) / page_height

    location_bonus = (
        1.0
        if center_y > 0.20
        else 0.65
    )

    aspect_score = min(
        aspect_ratio / 3.0,
        1.0,
    )

    area_score = min(
        area_fraction / 0.60,
        1.0,
    )

    structure_score = min(
        1.0,
        15 * horizontal_density
        + 15 * vertical_density
        + 5 * red_density,
    )

    density_score = min(
        ink_density / 0.12,
        1.0,
    )

    return (
        0.32 * area_score
        + 0.23 * aspect_score
        + 0.30 * structure_score
        + 0.10 * density_score
        + 0.05 * location_bonus
    )


def detect_structural_ecg_box(
    image: np.ndarray,
):
    """
    Detect the ECG region using lines, grid structure, and waveform ink.

    This is used only when the direct pink/red grid detector fails.
    """
    (
        detection_image,
        scale_x,
        scale_y,
    ) = resize_for_detection(
        image
    )

    (
        region_mask,
        binary,
        horizontal,
        vertical,
        red_grid,
    ) = candidate_region_mask(
        detection_image
    )

    contours, _ = cv2.findContours(
        region_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    page_height, page_width = (
        detection_image.shape[:2]
    )

    page_area = (
        page_height
        * page_width
    )

    candidates = []

    for contour in contours:
        (
            x,
            y,
            candidate_width,
            candidate_height,
        ) = cv2.boundingRect(
            contour
        )

        area_fraction = (
            candidate_width
            * candidate_height
            / page_area
        )

        aspect_ratio = (
            candidate_width
            / max(candidate_height, 1)
        )

        if (
            area_fraction
            < MIN_REGION_AREA_FRACTION
        ):
            continue

        if (
            area_fraction
            > MAX_REGION_AREA_FRACTION
        ):
            continue

        if (
            aspect_ratio
            < MIN_REGION_ASPECT_RATIO
        ):
            continue

        score = score_candidate_box(
            box=(
                x,
                y,
                candidate_width,
                candidate_height,
            ),
            image_shape=detection_image.shape,
            binary=binary,
            horizontal=horizontal,
            vertical=vertical,
            red_grid=red_grid,
        )

        candidates.append(
            {
                "box": (
                    x,
                    y,
                    candidate_width,
                    candidate_height,
                ),
                "score": score,
            }
        )

    if not candidates:
        return None, None, {
            "method": "none",
            "detection_image": detection_image,
            "region_mask": region_mask,
        }

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    best = candidates[0]

    (
        x,
        y,
        candidate_width,
        candidate_height,
    ) = best["box"]

    x1 = int(
        round(x * scale_x)
    )

    y1 = int(
        round(y * scale_y)
    )

    x2 = int(
        round(
            (x + candidate_width)
            * scale_x
        )
    )

    y2 = int(
        round(
            (y + candidate_height)
            * scale_y
        )
    )

    full_height, full_width = (
        image.shape[:2]
    )

    padding_x = int(
        full_width
        * CROP_PADDING_FRACTION
    )

    padding_y = int(
        full_height
        * CROP_PADDING_FRACTION
    )

    x1 = max(
        0,
        x1 - padding_x,
    )

    y1 = max(
        0,
        y1 - padding_y,
    )

    x2 = min(
        full_width,
        x2 + padding_x,
    )

    y2 = min(
        full_height,
        y2 + padding_y,
    )

    return (
        (x1, y1, x2, y2),
        best["score"],
        {
            "method": "structural",
            "detection_image": detection_image,
            "region_mask": region_mask,
            "candidates": candidates,
        },
    )


def detect_ecg_box(
    image: np.ndarray,
):
    """
    Detect the ECG plotting region.

    Priority:
      1. Pink/red ECG-grid detection.
      2. Structural detector fallback.
    """
    if USE_RED_GRID_DETECTION:
        (
            grid_box,
            grid_score,
            grid_mask,
        ) = detect_red_grid_box(
            image
        )

        if grid_box is not None:
            return (
                grid_box,
                grid_score,
                {
                    "method": "red-grid",
                    "grid_mask": grid_mask,
                },
            )

    return detect_structural_ecg_box(
        image
    )


def fallback_box(
    image: np.ndarray,
):
    """Create a conservative fallback crop after removing page borders."""
    height, width = image.shape[:2]

    x1 = int(
        width
        * FALLBACK_LEFT
    )

    x2 = int(
        width
        * FALLBACK_RIGHT
    )

    y1 = int(
        height
        * FALLBACK_TOP
    )

    y2 = int(
        height
        * FALLBACK_BOTTOM
    )

    return (
        x1,
        y1,
        x2,
        y2,
    )


# ============================================================
# OUTPUT AND DEBUGGING
# ============================================================

def draw_debug_box(
    image: np.ndarray,
    box,
    label: str,
) -> np.ndarray:
    """Draw the selected ECG crop on a copy of the rendered page."""
    debug = image.copy()

    x1, y1, x2, y2 = box

    cv2.rectangle(
        debug,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        thickness=max(
            3,
            image.shape[1] // 600,
        ),
    )

    cv2.putText(
        debug,
        label,
        (
            x1,
            max(
                35,
                y1 - 15,
            ),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(
            0.7,
            image.shape[1] / 2500,
        ),
        (0, 0, 255),
        max(
            2,
            image.shape[1] // 900,
        ),
        cv2.LINE_AA,
    )

    return debug


def save_image(
    path: Path,
    image: np.ndarray,
):
    """Save an image and raise an error if writing fails."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(path),
        image,
    )

    if not success:
        raise IOError(
            f"Could not save image: {path}"
        )


# ============================================================
# PROCESSING
# ============================================================

def process_pdf(pdf_path: Path):
    """
    Process a PDF only when it contains at most two pages.

    Returns:
        {
            "status": "constructed" | "skipped",
            "successful_crops": int,
            "number_of_pages": int,
        }
    """
    relative_path = pdf_path.relative_to(
        PDF_ROOT
    )

    document = fitz.open(
        pdf_path
    )

    number_of_pages = len(document)

    print()
    print(f"PDF: {relative_path}")
    print(f"Pages: {number_of_pages}")

    # Skip PDFs with more than two pages before rendering or creating outputs.
    if number_of_pages > 2:
        print(
            f"  SKIPPED: PDF contains {number_of_pages} pages. "
            "Only PDFs with one or two pages are processed."
        )

        document.close()

        return {
            "status": "skipped",
            "successful_crops": 0,
            "number_of_pages": number_of_pages,
        }

    pdf_output_directory = (
        OUTPUT_ROOT
        / relative_path.parent
        / safe_name(pdf_path.stem)
    )

    page_directory = (
        pdf_output_directory
        / "rendered_pages"
    )

    crop_directory = (
        pdf_output_directory
        / "ecg_crops"
    )

    debug_directory = (
        pdf_output_directory
        / "debug"
    )

    mask_directory = (
        pdf_output_directory
        / "grid_masks"
    )

    successful_crops = 0

    for page_index, page in enumerate(
        document
    ):
        page_number = page_index + 1

        rendered = render_pdf_page(
            page=page,
            dpi=RENDER_DPI,
        )

        (
            oriented,
            rotation,
            orientation_method,
        ) = orient_page(
            rendered
        )

        page_name = (
            f"page_{page_number:02d}"
        )

        if SAVE_RENDERED_PAGES:
            save_image(
                page_directory
                / f"{page_name}.png",
                oriented,
            )

        (
            box,
            score,
            detection_info,
        ) = detect_ecg_box(
            oriented
        )

        detection_method = (
            detection_info.get(
                "method",
                "unknown",
            )
        )

        if (
            SAVE_GRID_MASKS
            and "grid_mask" in detection_info
        ):
            save_image(
                mask_directory
                / f"{page_name}_grid_mask.png",
                detection_info["grid_mask"],
            )

        used_fallback = False

        if (
            box is None
            and SAVE_FALLBACK_CROP
        ):
            box = fallback_box(
                oriented
            )

            used_fallback = True
            detection_method = "fallback"

        if box is None:
            print(
                f"  Page {page_number}: "
                f"no ECG region detected; "
                f"rotation={rotation}° "
                f"({orientation_method})"
            )
            continue

        x1, y1, x2, y2 = box

        cropped = oriented[
            y1:y2,
            x1:x2,
        ]

        if cropped.size == 0:
            print(
                f"  Page {page_number}: "
                "empty crop, skipped"
            )
            continue

        if used_fallback:
            suffix = "fallback"
            debug_label = "Fallback crop"
        else:
            suffix = detection_method

            debug_label = (
                f"ECG crop, "
                f"method={detection_method}, "
                f"score={score:.3f}"
            )

        crop_path = (
            crop_directory
            / (
                f"{page_name}_"
                f"ecg_{suffix}.png"
            )
        )

        save_image(
            crop_path,
            cropped,
        )

        if SAVE_DEBUG_IMAGES:
            debug_image = draw_debug_box(
                image=oriented,
                box=box,
                label=debug_label,
            )

            save_image(
                debug_directory
                / (
                    f"{page_name}_"
                    f"detection_{suffix}.png"
                ),
                debug_image,
            )

        successful_crops += 1

        score_text = (
            "fallback"
            if used_fallback
            else f"{score:.3f}"
        )

        print(
            f"  Page {page_number}: "
            f"saved {crop_path.name}; "
            f"rotation={rotation}° "
            f"({orientation_method}); "
            f"method={detection_method}; "
            f"score={score_text}"
        )

    document.close()

    status = (
        "constructed"
        if successful_crops > 0
        else "no_crops"
    )

    return {
        "status": status,
        "successful_crops": successful_crops,
        "number_of_pages": number_of_pages,
    }


def main():
    if not PDF_ROOT.exists():
        raise FileNotFoundError(
            f"PDF directory was not found: "
            f"{PDF_ROOT}"
        )

    pdf_files = sorted(
        path
        for path in PDF_ROOT.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_PDF_SUFFIXES
        )
    )

    if not pdf_files:
        raise RuntimeError(
            f"No PDF files were found under: "
            f"{PDF_ROOT}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("ECG PDF REGION EXTRACTION")
    print("=" * 80)
    print(f"PDF root:       {PDF_ROOT}")
    print(f"Output root:    {OUTPUT_ROOT}")
    print(f"Number of PDFs: {len(pdf_files)}")
    print(f"Rendering DPI:  {RENDER_DPI}")

    total_crops = 0
    constructed_files = 0
    skipped_more_than_two_pages = 0
    no_crop_files = 0
    failed_pdfs = []

    for pdf_index, pdf_path in enumerate(
        pdf_files,
        start=1,
    ):
        print()
        print(
            f"[{pdf_index}/{len(pdf_files)}]"
        )

        try:
            result = process_pdf(
                pdf_path
            )

            total_crops += result[
                "successful_crops"
            ]

            if result["status"] == "constructed":
                constructed_files += 1

            elif result["status"] == "skipped":
                skipped_more_than_two_pages += 1

            elif result["status"] == "no_crops":
                no_crop_files += 1

        except Exception as error:
            failed_pdfs.append(
                (
                    str(pdf_path),
                    str(error),
                )
            )

            print(
                f"  ERROR: {error}"
            )

    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(
        f"PDF files discovered:              "
        f"{len(pdf_files)}"
    )

    print(
        f"Files constructed:                 "
        f"{constructed_files}"
    )

    print(
        f"Skipped because pages > 2:         "
        f"{skipped_more_than_two_pages}"
    )

    print(
        f"Files with no successful crop:     "
        f"{no_crop_files}"
    )

    print(
        f"ECG crop images saved:             "
        f"{total_crops}"
    )

    print(
        f"Failed PDFs:                       "
        f"{len(failed_pdfs)}"
    )

    print(
        f"Outputs saved under:               "
        f"{OUTPUT_ROOT}"
    )

    if failed_pdfs:
        failure_file = (
            OUTPUT_ROOT
            / "failed_pdfs.txt"
        )

        with failure_file.open(
            "w",
            encoding="utf-8",
        ) as file:
            for (
                failed_pdf_path,
                error,
            ) in failed_pdfs:
                file.write(
                    f"{failed_pdf_path}\n"
                )

                file.write(
                    f"ERROR: {error}\n\n"
                )

        print(
            f"Failure details:                   "
            f"{failure_file}"
        )


if __name__ == "__main__":
    main()