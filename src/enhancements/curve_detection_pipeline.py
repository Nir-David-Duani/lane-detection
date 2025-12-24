"""
Curve Detection Pipeline - Helper Library
==========================================
This module provides building blocks for curve lane detection.

The module includes:
1. ROI masking and geometry helpers
2. Color and gradient masking (HSV, HLS, CLAHE)
3. Edge detection and Hough line detection
4. Polynomial fitting (linear and quadratic)
5. Lane polygon construction and overlay
6. Mask scoring and selection
7. Geometry enforcement for lane width constraints

Design goals:
- Keep curve detection self-contained: curve code should import from THIS module only
- Preserve the exact working logic (masking, gating, Hough proximity selection, polynomial fitting, overlay)
- Provide small building blocks that note where they came from (debug-approved logic vs. migrated shared helpers)

Notes:
- This file intentionally contains many helpers that are not called internally. They are imported by
  `enhancements/curve_detection.py` and `debug.py`.
- When cleaning/formatting this file, do NOT change algorithmic logic. Only refactor for readability,
  typing, and documentation.
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Sequence, Any

import cv2
import numpy as np

# ============================================================
#              CURVE DEFAULTS (SINGLE SOURCE OF TRUTH)
# ============================================================

from dataclasses import dataclass


@dataclass(frozen=True)
class CurveDefaults:
    """Default parameters for the curve detection runner.

    IMPORTANT:
    - These defaults must match the values used by `enhancements/curve_detection.py`.
    - Debug and other callers should prefer fetching these defaults from here to avoid drift.
    - Any geometry enforcement defaults must remain OFF by default to preserve production behavior.

    Notes:
    - Ratios are in [0, 1].
    - Pixel distances are in pixels.
    """

    # ROI geometry
    top_y_ratio: float = 0.62
    left_bottom_ratio: float = 0.10
    right_bottom_ratio: float = 0.90
    top_left_x_ratio: float = 0.45
    top_right_x_ratio: float = 0.55

    # Overlay
    overlay_top_y_ratio: float = 0.65
    lane_fill_alpha: float = 0.30

    # Preprocessing
    clahe_clip: float = 2.0
    clahe_grid: int = 8

    # Mask gating / Hough
    merge_grad_dist_px: int = 12
    hough_dist_px: float = 10.0

    # Edges for Hough (kept here so debug and runner match out of the box)
    canny1: int = 30
    canny2: int = 100
    gate_dilate_px: int = 15

    # Optional geometry enforcement (OFF by default - debug-only unless explicitly enabled)
    geom_enforce_bottom_width: bool = False
    geom_bottom_band_ratio: float = 0.30
    geom_margin_ratio: float = 0.02
    geom_min_width_ratio: float = 0.20
    geom_max_width_ratio: Optional[float] = 0.40
    geom_anchor_weight: float = 0.0


def get_curve_defaults() -> CurveDefaults:
    """
    Return default curve parameters.
    
    Centralizes defaults so debug and the curve runner can stay consistent.
    
    Returns:
        CurveDefaults: Default parameter dataclass instance
    """
    return CurveDefaults()


def _mask_from_pts(h: int, w: int, pts_2d: np.ndarray) -> np.ndarray:
    """
    Return a filled polygon mask (0/255) for polygon points shaped (N,2) in (x,y).
    
    Args:
        h: Image height
        w: Image width
        pts_2d: Polygon points array shaped (N, 2) in (x, y)
    
    Returns:
        Binary mask (0/255) with filled polygon
    """
    m = np.zeros((int(h), int(w)), dtype=np.uint8)
    cv2.fillPoly(m, [pts_2d.astype(np.int32)], 255)
    return m


def build_alt_rois_from_main_roi(
    frame_bgr: np.ndarray,
    *,
    roi_pts: np.ndarray,
    top_y_ratio: float,
    left_bottom_ratio: float,
    right_bottom_ratio: float,
    top_left_x_ratio: float,
    top_right_x_ratio: float,
    alt_top_offset_ratio_of_roi_height: float = 0.02,
    ext_ratio_of_top_width: float = 0.50,
    ext_min_ratio_of_w: float = 0.08,
    ext_min_px: int = 45,
    shift_ratio_of_top_width: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build asymmetric ALT ROIs (left and right) derived from the main ROI geometry.

    This matches the logic used in the current debug sampler and curve runner:
    - Same bottom base as the main ROI.
    - The top edge is moved slightly upward (by a fraction of ROI trapezoid height).
    - The top base is extended and shifted:
      * ALT L extends to the left and shifts the top base left.
      * ALT R extends to the right and shifts the top base right.

    Returns:
      roi_alt_L_bgr, roi_alt_R_bgr: BGR frames masked by the ALT L/R polygons.

    Notes:
    - This helper intentionally returns only the masked images because that is what the
      color mask builders consume.
    """
    h, w = frame_bgr.shape[:2]

    # Use ROI top derived from the actual ROI polygon (more robust than re-deriving from ratios).
    y_top_roi = roi_top_y_from_pts(roi_pts, int(h))
    y_bot = int(h - 1)
    roi_height = max(1, (y_bot - int(y_top_roi)))

    # Move the top slightly upward.
    y_top_alt = int(round(float(y_top_roi) - float(alt_top_offset_ratio_of_roi_height) * float(roi_height)))
    y_top_alt = int(np.clip(y_top_alt, 0, int(h - 2)))

    # Main ROI x coordinates (derived from ratios).
    x_lb = int(float(w) * float(left_bottom_ratio))
    x_rb = int(float(w) * float(right_bottom_ratio))
    x_tl = int(float(w) * float(top_left_x_ratio))
    x_tr = int(float(w) * float(top_right_x_ratio))

    top_width = float(max(1, (x_tr - x_tl)))

    # Extend top base by a noticeable amount (match debug/runner behavior).
    ext = int(max(round(float(ext_ratio_of_top_width) * top_width), round(float(ext_min_ratio_of_w) * float(w)), int(ext_min_px)))

    # Shift entire top base sideways.
    shift = int(max(round(float(shift_ratio_of_top_width) * top_width), 0))

    # ALT L: extend left and shift left
    x_tl_L = int(np.clip(int(x_tl - ext - shift), 0, int(w - 1)))
    x_tr_L = int(np.clip(int(x_tr - shift), 0, int(w - 1)))
    if x_tl_L >= x_tr_L:
        x_tl_L = int(max(0, int(x_tr_L - 1)))

    pts_alt_L = np.array(
        [[x_lb, y_bot], [x_rb, y_bot], [x_tr_L, y_top_alt], [x_tl_L, y_top_alt]],
        dtype=np.int32,
    )
    mask_alt_L = _mask_from_pts(int(h), int(w), pts_alt_L)
    roi_alt_L_bgr = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask_alt_L)

    # ALT R: extend right and shift right
    x_tl_R = int(np.clip(int(x_tl + shift), 0, int(w - 1)))
    x_tr_R = int(np.clip(int(x_tr + ext + shift), 0, int(w - 1)))
    if x_tl_R >= x_tr_R:
        x_tr_R = int(min(int(w - 1), int(x_tl_R + 1)))

    pts_alt_R = np.array(
        [[x_lb, y_bot], [x_rb, y_bot], [x_tr_R, y_top_alt], [x_tl_R, y_top_alt]],
        dtype=np.int32,
    )
    mask_alt_R = _mask_from_pts(int(h), int(w), pts_alt_R)
    roi_alt_R_bgr = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask_alt_R)

    return roi_alt_L_bgr, roi_alt_R_bgr


def resolve_curve_params(overrides: Optional[Dict[str, Any]] = None) -> CurveDefaults:
    """
    Return defaults with optional overrides applied.
    
    This is a small convenience so CLI tools can override a subset of fields while
    keeping all other values consistent.
    
    Args:
        overrides: Optional dictionary of parameter overrides
    
    Returns:
        CurveDefaults: Parameter dataclass with overrides applied
    
    Example:
        params = resolve_curve_params({"hough_dist_px": 14.0})
    """
    base = get_curve_defaults()
    if not overrides:
        return base

    data = base.__dict__.copy()
    for k, v in overrides.items():
        if k in data and v is not None:
            data[k] = v
    return CurveDefaults(**data)

# Type aliases for HSV bounds (for documentation only)
HSVBound = Sequence[int]


# ============================================================
#              LEGACY SHARED HELPERS
# ============================================================
# Migrated from pipeline.py

def apply_roi_mask(
    frame: np.ndarray,
    top_y_ratio: float,
    left_bottom_ratio: float,
    right_bottom_ratio: float,
    top_left_x_ratio: float,
    top_right_x_ratio: float,
    bottom_y_ratio: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a trapezoidal Region of Interest (ROI) mask.

    This is a geometric helper that uses normalized ratios (0..1) along the image
    width/height. It does not assume any specific camera or road layout - those
    are encoded only in the ratios passed in by the caller.

    Args:
        frame: Input BGR image.
        top_y_ratio: Vertical position of trapezoid top (0..1).
        left_bottom_ratio: Left edge position at bottom (0..1).
        right_bottom_ratio: Right edge position at bottom (0..1).
        top_left_x_ratio: Left edge position at top (0..1).
        top_right_x_ratio: Right edge position at top (0..1).
        bottom_y_ratio: Vertical position of trapezoid bottom (0..1, default 1.0).

    Returns:
        masked: Image with ROI applied (black outside ROI).
        pts: Polygon points array (np.int32) for visualization.
        Note: This is the same trapezoid mask logic as the original straight-line pipeline (migrated here to remove dependencies).
    """
    h, w = frame.shape[:2]

    top_y = int(h * float(top_y_ratio))
    bottom_y = int(h * float(bottom_y_ratio))

    # Safety clamp
    top_y = max(0, min(h - 1, top_y))
    bottom_y = max(0, min(h - 1, bottom_y))

    pts = np.array(
        [[
            (int(w * float(left_bottom_ratio)), bottom_y),
            (int(w * float(right_bottom_ratio)), bottom_y),
            (int(w * float(top_right_x_ratio)), top_y),
            (int(w * float(top_left_x_ratio)), top_y),
        ]],
        dtype=np.int32,
    )

    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, pts, (255, 255, 255))
    masked = cv2.bitwise_and(frame, mask)

    return masked, pts


def apply_color_threshold(
    frame_roi: np.ndarray,
    white_lower: HSVBound,
    white_upper: HSVBound,
    yellow_lower: HSVBound,
    yellow_upper: HSVBound,
    morph_kernel_size: int,
    morph_open_iter: int,
    morph_close_iter: int,
    morph_dilate_iter: int,
) -> np.ndarray:
    """This is the exact HSV threshold + morphology helper migrated from pipeline.py.

    Generic HSV color thresholding + morphology.

    Args:
        frame_roi: Input BGR image (typically after ROI masking).
        white_lower/white_upper: HSV bounds for white.
        yellow_lower/yellow_upper: HSV bounds for yellow.
        morph_kernel_size: Size of square kernel for morphology ops.
        morph_open_iter: Iterations for opening.
        morph_close_iter: Iterations for closing.
        morph_dilate_iter: Iterations for dilation.

    Returns:
        mask: Binary mask (255 = foreground, 0 = background).
    """
    hsv = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2HSV)

    lower_white = np.array(white_lower, dtype=np.uint8)
    upper_white = np.array(white_upper, dtype=np.uint8)
    lower_yellow = np.array(yellow_lower, dtype=np.uint8)
    upper_yellow = np.array(yellow_upper, dtype=np.uint8)

    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    mask = cv2.bitwise_or(mask_white, mask_yellow)

    kernel = np.ones((int(morph_kernel_size), int(morph_kernel_size)), np.uint8)
    if int(morph_open_iter) > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=int(morph_open_iter))
    if int(morph_close_iter) > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=int(morph_close_iter))
    if int(morph_dilate_iter) > 0:
        mask = cv2.dilate(mask, kernel, iterations=int(morph_dilate_iter))

    return mask


def apply_canny(mask: np.ndarray, low_threshold: int, high_threshold: int, blur_kernel: int) -> np.ndarray:
    """Generic Canny edge detection with pre-blur.

    Args:
        mask: Binary or grayscale image (0-255).
        low_threshold: Lower threshold for hysteresis.
        high_threshold: Upper threshold for hysteresis.
        blur_kernel: Gaussian blur kernel size (must be odd).

    Returns:
        edges: Binary edge map (255 = edge, 0 = no edge).
    """
    k = int(blur_kernel)
    if k % 2 == 0:
        k += 1
    k = max(1, k)

    blurred = cv2.GaussianBlur(mask, (k, k), 0)
    edges = cv2.Canny(blurred, int(low_threshold), int(high_threshold))
    return edges


def detect_lines_hough(
    edges: np.ndarray,
    rho: float,
    theta: float,
    threshold: int,
    min_line_length: int,
    max_line_gap: int,
) -> Optional[np.ndarray]:
    """Generic straight-line detection using Probabilistic Hough Transform.

    Args:
        edges: Binary edge map from Canny detection.
        rho: Distance resolution in pixels.
        theta: Angle resolution in radians.
        threshold: Minimum number of intersections to detect a line.
        min_line_length: Minimum length of line segment in pixels.
        max_line_gap: Maximum gap between points to be considered same line.

    Returns:
        lines: Array of line segments [[x1, y1, x2, y2], ...], or None if no lines.
    """
    return cv2.HoughLinesP(
        edges,
        rho=float(rho),
        theta=float(theta),
        threshold=int(threshold),
        minLineLength=int(min_line_length),
        maxLineGap=int(max_line_gap),
    )


# ============================================================
#              SMALL UTILITIES
# ============================================================

def to_mask_01(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Normalize any non-empty mask to a binary 0/255 uint8 2D mask."""
    if mask is None:
        return None
    m = mask
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    m = ((m > 0).astype(np.uint8)) * 255
    return m




# ============================================================
#              COLOR / GRADIENT MASKS
# ============================================================

def mask_hsv_curve_tuned(
    frame_bgr: np.ndarray,
    *,
    # White tuning (reduce noise): require high V and low S
    white_v_min: int = 190,
    white_s_max: int = 115,
    # Yellow tuning
    yellow_h_lo: int = 15,
    yellow_h_hi: int = 45,
    yellow_s_min: int = 80,
    yellow_v_min: int = 80,
    # Morphology
    open_iter: int = 1,
    close_iter: int = 2,
    dilate_iter: int = 1,
) -> np.ndarray:
    """Binary (0/255) lane mask using HSV - curve-detection dedicated copy.

    Purpose:
    - Keep straight-lines pipeline untouched.
    - Provide a curve-only HSV mask that can be tuned to reduce white noise.

    White policy (noise reduction):
    - Keep pixels that are fairly bright (V >= white_v_min)
    - and not overly saturated (S <= white_s_max)

    Yellow policy:
    - Hue in [yellow_h_lo, yellow_h_hi], S >= yellow_s_min, V >= yellow_v_min
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # White: bright + low saturation
    white_v = cv2.inRange(V, int(white_v_min), 255)
    white_s = cv2.inRange(S, 0, int(white_s_max))
    white = cv2.bitwise_and(white_v, white_s)

    # Yellow: hue window + strong saturation + sufficient brightness
    yellow_h = cv2.inRange(H, int(yellow_h_lo), int(yellow_h_hi))
    yellow_s = cv2.inRange(S, int(yellow_s_min), 255)
    yellow_v = cv2.inRange(V, int(yellow_v_min), 255)
    yellow = cv2.bitwise_and(yellow_h, cv2.bitwise_and(yellow_s, yellow_v))

    mask = cv2.bitwise_or(white, yellow)

    # Light cleanup (keep dashed lanes)
    k = np.ones((3, 3), dtype=np.uint8)
    if int(open_iter) > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=int(open_iter))
    if int(close_iter) > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=int(close_iter))
    if int(dilate_iter) > 0:
        mask = cv2.dilate(mask, k, iterations=int(dilate_iter))

    return to_mask_01(mask)

def mask_hls_clahe(frame_bgr: np.ndarray, clahe_clip: float = 2.0, clahe_grid: int = 8) -> np.ndarray:
    """Binary (0/255) lane mask using HLS + CLAHE on L.

    This is copied from the working debug pipeline (curve debug).
    """
    hls = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HLS)
    H, L, S = cv2.split(hls)

    clahe = cv2.createCLAHE(
        clipLimit=float(clahe_clip),
        tileGridSize=(int(clahe_grid), int(clahe_grid)),
    )
    L_eq = clahe.apply(L)

    white = cv2.inRange(L_eq, 200, 255)

    yellow_h = cv2.inRange(H, 15, 45)
    yellow_s = cv2.inRange(S, 80, 255)
    yellow = cv2.bitwise_and(yellow_h, yellow_s)

    mask = cv2.bitwise_or(white, yellow)

    k = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.dilate(mask, k, iterations=2)
    return mask


def mask_gradient_edge(frame_bgr: np.ndarray, clahe_clip: float = 2.0, clahe_grid: int = 8) -> np.ndarray:
    """Binary (0/255) mask using grayscale contrast + Canny edges."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=float(clahe_clip),
        tileGridSize=(int(clahe_grid), int(clahe_grid)),
    )
    g = clahe.apply(gray)

    g_blur = cv2.GaussianBlur(g, (5, 5), 0)
    edges = cv2.Canny(g_blur, 50, 150)

    k = np.ones((3, 3), dtype=np.uint8)
    edges = cv2.dilate(edges, k, iterations=1)
    return edges


def filter_grad_by_slope(grad_mask: np.ndarray, min_deg: float = 20.0, max_deg: float = 75.0) -> Optional[np.ndarray]:
    """Keep only gradient components whose orientation matches lane-like slopes."""
    g = to_mask_01(grad_mask)
    if g is None:
        return None

    out = np.zeros_like(g)
    num, labels, stats, _ = cv2.connectedComponentsWithStats((g > 0).astype(np.uint8), connectivity=8)

    for i in range(1, num):
        area = stats[i, 4]
        if area < 50:
            continue

        ys, xs = np.where(labels == i)
        if len(xs) < 2:
            continue

        vx, vy = cv2.fitLine(np.column_stack((xs, ys)), cv2.DIST_L2, 0, 0.01, 0.01)[:2]
        vx_f = float(vx[0]) if isinstance(vx, np.ndarray) else float(vx)
        vy_f = float(vy[0]) if isinstance(vy, np.ndarray) else float(vy)
        angle = abs(float(np.degrees(np.arctan2(vy_f, vx_f))))
        if float(min_deg) <= angle <= float(max_deg):
            out[labels == i] = 255

    return out


def clean_mask_light(mask_01: np.ndarray) -> np.ndarray:
    """Light cleaning - preserve dashed lanes."""
    m = to_mask_01(mask_01)
    if m is None:
        # Defensive: keep downstream code simple (always return a mask).
        return np.zeros(mask_01.shape[:2], dtype=np.uint8)

    k3 = np.ones((3, 3), dtype=np.uint8)
    k5 = np.ones((5, 5), dtype=np.uint8)

    # Close small gaps, but avoid heavy dilation that would merge lanes
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k5, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k3, iterations=1)
    return m


# ============================================================
#              BLOB SUPPRESSION (VEHICLE-IN-LANE)
# ============================================================


def suppress_center_blob_components(
    mask_01: np.ndarray,
    *,
    y_bottom_ratio: float = 0.92,
    center_band_ratio: float = 0.46,
    min_area: int = 300,
    min_width_ratio: float = 0.10,
    min_height_ratio: float = 0.04,
    max_height_ratio: float = 0.55,
) -> np.ndarray:
    """Remove large center-ish blobs (often the rear of a car) from a binary lane mask.

    Strategy:
    - Work only on the top part of the ROI (up to y_bottom_ratio). The bottom is kept intact.
    - Remove connected components whose centroid lies inside a central x-band,
      and whose bounding box is sufficiently large.

    Returns a 0/255 uint8 mask.
    """
    m = to_mask_01(mask_01)
    if m is None:
        return mask_01

    h, w = m.shape[:2]
    y1 = int(np.clip(int(h * float(y_bottom_ratio)), 0, h))
    if y1 <= 10:
        return m

    xc = 0.5 * float(w)
    half_band = 0.5 * float(center_band_ratio) * float(w)
    x_lo = int(np.clip(int(xc - half_band), 0, w - 1))
    x_hi = int(np.clip(int(xc + half_band), 0, w - 1))

    band = m[:y1, :]
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (band > 0).astype(np.uint8), connectivity=8
    )

    out = m.copy()
    min_w = float(min_width_ratio) * float(w)
    min_h = float(min_height_ratio) * float(h)
    max_h = float(max_height_ratio) * float(h)

    for i in range(1, int(num)):
        x, y, bw, bh, area = stats[i]
        if int(area) < int(min_area):
            continue

        cx, _cy = centroids[i]
        if not (float(x_lo) <= float(cx) <= float(x_hi)):
            continue

        # Large-ish, center-ish blob
        if float(bw) < min_w:
            continue
        if float(bh) < min_h:
            continue
        if float(bh) > max_h:
            continue

        # bbox overlaps center band
        if (x + bw) < x_lo or x > x_hi:
            continue

        out[:y1, :][labels == i] = 0

    return out


# Helper to suppress center blob-like edge components (vehicle edges) directly in edge image
def suppress_center_blob_edges(
    edges_01: np.ndarray,
    *,
    y_bottom_ratio: float = 0.92,
    center_band_ratio: float = 0.46,
    min_area: int = 250,
    min_width_ratio: float = 0.10,
    min_height_ratio: float = 0.03,
    max_height_ratio: float = 0.60,
) -> np.ndarray:
    """Remove large center-ish edge components (often a vehicle) from an edge image.

    This operates on the *edge* image directly (not the mask). It removes connected components
    whose centroid lies in a central x-band and whose bbox is large enough.

    Returns a 0/255 uint8 edge image.
    """
    e = to_mask_01(edges_01)
    if e is None:
        return edges_01

    h, w = e.shape[:2]
    y1 = int(np.clip(int(h * float(y_bottom_ratio)), 0, h))
    if y1 <= 10:
        return e

    xc = 0.5 * float(w)
    half_band = 0.5 * float(center_band_ratio) * float(w)
    x_lo = int(np.clip(int(xc - half_band), 0, w - 1))
    x_hi = int(np.clip(int(xc + half_band), 0, w - 1))

    band = e[:y1, :]
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (band > 0).astype(np.uint8), connectivity=8
    )

    out = e.copy()
    min_w = float(min_width_ratio) * float(w)
    min_h = float(min_height_ratio) * float(h)
    max_h = float(max_height_ratio) * float(h)

    for i in range(1, int(num)):
        x, y, bw, bh, area = stats[i]
        if int(area) < int(min_area):
            continue

        cx, _cy = centroids[i]
        if not (float(x_lo) <= float(cx) <= float(x_hi)):
            continue

        if float(bw) < min_w:
            continue
        if float(bh) < min_h:
            continue
        if float(bh) > max_h:
            continue

        # bbox overlaps center band
        if (x + bw) < x_lo or x > x_hi:
            continue

        out[:y1, :][labels == i] = 0

    return out


# ============================================================
#              MERGE LOGIC (DEBUG-APPROVED)
# ============================================================

def merge_grad_with_color_for_score(
    color_mask: Optional[np.ndarray],
    grad_mask: Optional[np.ndarray],
    max_dist_px: int,
    min_grad_nz_for_gating: int = 200,
) -> Optional[np.ndarray]:
    """Merge COLOR into GRAD using dilated-GRAD gating (score merge).

    merged = GRAD ∪ (COLOR ∩ dilate(GRAD, r))

    This is the exact logic from debug (curve sampler).
    """
    if color_mask is None and grad_mask is None:
        return None
    if color_mask is None:
        return to_mask_01(grad_mask)
    if grad_mask is None:
        return to_mask_01(color_mask)

    c = to_mask_01(color_mask)
    g = to_mask_01(grad_mask)

    if c is None:
        return g
    if g is None:
        return c

    # If GRAD is empty or too sparse, do NOT gate color by GRAD.
    # This prevents the MERGED mask from collapsing when GRAD is poor.
    g_nz = int(cv2.countNonZero(g))
    if g_nz == 0:
        return c
    if g_nz < int(min_grad_nz_for_gating):
        return cv2.bitwise_or(c, g)

    r = int(max(0, int(max_dist_px)))
    if r > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        g_gate = cv2.dilate(g, k, iterations=1)
    else:
        g_gate = g

    merged = g.copy()
    merged[(c > 0) & (g_gate > 0)] = 255
    return merged


def build_corridor_from_mask(mask_01: Optional[np.ndarray], dilate_px: int) -> Optional[np.ndarray]:
    """Build a corridor mask (0/255) by dilating an input mask."""
    m = to_mask_01(mask_01)
    if m is None:
        return None

    r = int(max(0, int(dilate_px)))
    if r <= 0:
        return m

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(m, k, iterations=1)


def merge_masks_for_hough_union(
    color_mask: Optional[np.ndarray],
    grad_mask: Optional[np.ndarray],
    corridor_dilate_px: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Union-like merge for Hough only (debug-approved).

    Returns:
      - merged_union: (COLOR ∪ GRAD) restricted to a dilated corridor of (COLOR ∪ GRAD)
      - corridor: the corridor mask used for gating (0/255)
    """
    if color_mask is None and grad_mask is None:
        return None, None

    c = to_mask_01(color_mask) if color_mask is not None else None
    g = to_mask_01(grad_mask) if grad_mask is not None else None

    if c is None:
        base = g
    elif g is None:
        base = c
    else:
        base = cv2.bitwise_or(c, g)

    corridor = build_corridor_from_mask(base, dilate_px=int(corridor_dilate_px))
    if corridor is None or base is None:
        return base, corridor

    merged_union = cv2.bitwise_and(base, base, mask=corridor)
    return merged_union, corridor


# ============================================================
#              HOUGH EDGES HELPER (DEBUG-APPROVED)
# ============================================================

def edges_for_hough(
    roi_bgr: np.ndarray,
    mask_clean_01: np.ndarray,
    y_top_keep: int,
    clahe_clip: float,
    clahe_grid: int,
    canny1: int = 50,
    canny2: int = 150,
    gate_dilate_px: int = 5,
) -> np.ndarray:
    """Edge image for Hough (debug-approved).

    Important policy (from debug):
    - do NOT hard-AND with the exact mask (kills dashed lanes)
    - gate by a dilated corridor around the mask
    """
    h, _w = roi_bgr.shape[:2]

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=float(clahe_clip),
        tileGridSize=(int(clahe_grid), int(clahe_grid)),
    )
    g = clahe.apply(gray)
    g_blur = cv2.GaussianBlur(g, (5, 5), 0)
    edges = cv2.Canny(g_blur, int(canny1), int(canny2))

    m = to_mask_01(mask_clean_01)
    if m is None:
        m = np.zeros((h, roi_bgr.shape[1]), dtype=np.uint8)

    if int(gate_dilate_px) > 0:
        r = int(gate_dilate_px)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        m = cv2.dilate(m, k, iterations=1)

    edges = cv2.bitwise_and(edges, edges, mask=m)

    y_top_keep = int(np.clip(int(y_top_keep), 0, h))
    edges[:y_top_keep, :] = 0

    return ((edges > 0).astype(np.uint8)) * 255


# ============================================================
#              ROI HELPERS
# ============================================================

def roi_mask_full_frame(
    frame_shape: Tuple[int, int, int],
    *,
    top_y_ratio: float,
    left_bottom_ratio: float,
    right_bottom_ratio: float,
    top_left_x_ratio: float,
    top_right_x_ratio: float,
) -> np.ndarray:
    """Binary ROI mask (0/255) for a trapezoid over the full frame."""
    h, w = frame_shape[:2]
    top_y = int(h * float(top_y_ratio))
    top_y = max(0, min(h - 1, top_y))

    pts = np.array(
        [[
            (int(w * float(left_bottom_ratio)), h - 1),
            (int(w * float(right_bottom_ratio)), h - 1),
            (int(w * float(top_right_x_ratio)), top_y),
            (int(w * float(top_left_x_ratio)), top_y),
        ]],
        dtype=np.int32,
    )

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, pts, 255)
    return mask


def alt_roi_rect_plus_trapezoid(
    frame_bgr: np.ndarray,
    *,
    alt_top_y_ratio: float,
    alt_bottom_y_ratio: float,
    alt_top_left_x_ratio: float,
    alt_top_right_x_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """ALT ROI = rectangle (bottom -> bottom_y) + trapezoid (bottom_y -> top_y)."""
    h, w = frame_bgr.shape[:2]

    top_y = int(h * float(alt_top_y_ratio))
    bottom_y = int(h * float(alt_bottom_y_ratio))

    top_y = max(0, min(h - 1, top_y))
    bottom_y = max(0, min(h - 1, bottom_y))

    mask = np.zeros((h, w), dtype=np.uint8)

    rect_pts = np.array(
        [[
            (0, h - 1),
            (w - 1, h - 1),
            (w - 1, bottom_y),
            (0, bottom_y),
        ]],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, rect_pts, 255)

    trap_pts = np.array(
        [[
            (0, bottom_y),
            (w - 1, bottom_y),
            (int(w * float(alt_top_right_x_ratio)), top_y),
            (int(w * float(alt_top_left_x_ratio)), top_y),
        ]],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, trap_pts, 255)

    masked = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)
    return masked, mask


def roi_top_y_from_pts(roi_pts: np.ndarray, h: int) -> int:
    """Top y of the ROI trapezoid (horizontal apex height), not the apex point."""
    try:
        pts = roi_pts
        if pts.ndim == 3:
            pts = pts[0]
        y_top = int(min(float(pts[2][1]), float(pts[3][1])))
        return int(np.clip(y_top, 0, h - 2))
    except Exception:
        return int(np.clip(0.60 * h, 0, h - 2))



# ============================================================
#              OPTIONAL GEOMETRY ENFORCEMENT (OFF BY DEFAULT)
# ============================================================

def enforce_min_lane_width_bottom_band(
    xL: np.ndarray,
    xR: np.ndarray,
    y_vals: np.ndarray,
    *,
    roi_pts: Optional[np.ndarray] = None,
    h: int,
    w: int,
    y_top_draw: Optional[int] = None,
    enabled: bool = False,
    bottom_band_ratio: float = 0.30,
    margin_ratio: float = 0.02,
    min_width_ratio: float = 0.20,
    max_width_ratio: Optional[float] = None,
    xsL_sel: Optional[np.ndarray] = None,
    ysL_sel: Optional[np.ndarray] = None,
    xsR_sel: Optional[np.ndarray] = None,
    ysR_sel: Optional[np.ndarray] = None,
    anchor_weight: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray]:
    """Optionally enforce plausible lane geometry on the *bottom band* only.

    This helper is designed to be SAFE for production when disabled:
    - If enabled=False, it returns inputs unchanged.

    Intended usage (debug experimentation first):
    - After you compute/evaluate xL(y), xR(y) on a fixed set of sample heights (K samples),
      call this helper to prevent impossible geometry near the bottom when one side is missing.

    What it enforces (only for y in the bottom band of the ROI):
    - A minimum lane width (in pixels) derived from `min_width_ratio * w`.
    - An optional maximum width (if max_width_ratio is provided).

    Adjustment policy:
    - When the gap is too small, expand symmetrically around the midline (keep center stable).
    - When the gap is too large (optional), shrink symmetrically around the midline.

    Notes:
    - This is NOT a "mirror left from right" rule. It only constrains width.
    - Works on arrays xL/xR already aligned by y_vals (same length).
    """
    if not enabled:
        return xL, xR

    if xL is None or xR is None or y_vals is None:
        return xL, xR

    xL_in = xL
    xR_in = xR

    xL2 = np.asarray(xL, dtype=np.float32).copy()
    xR2 = np.asarray(xR, dtype=np.float32).copy()
    y = np.asarray(y_vals, dtype=np.int32)

    # Some callers provide y-values in "sample index" space (e.g., 0..K-1) rather than
    # absolute pixel y-coordinates. In that case, map indices to pixel y along the ROI span
    # so bottom-band logic works as intended.
    y_pix = y
    try:
        n_in = int(y.size)
        if n_in >= 2:
            y_min_in = int(np.min(y))
            y_max_in = int(np.max(y))

            # Detect index-space y robustly:
            # - starts at 0
            # - values stay within ~[0, n-1] (allow a small slack for rounding)
            # - y is non-decreasing (linspace rounding can create duplicates)
            #
            # This avoids the systematic "no effect" case where y is 0..K-ish while
            # the image height is much larger (e.g., K=480, h=1080), so the bottom-band
            # predicate y_pix >= y_band_top never triggers.
            slack = 2
            nondecreasing = bool(np.all(y[1:] >= y[:-1]))
            y_is_index_space = bool(
                (y_min_in == 0)
                and (y_max_in <= (n_in - 1 + slack))
                and nondecreasing
            )
        else:
            y_is_index_space = False
    except Exception:
        y_is_index_space = False

    if xL2.size == 0 or xR2.size == 0 or y.size == 0:
        return xL, xR

    # Determine ROI vertical span and bottom-band threshold.
    if roi_pts is not None:
        y_top_roi = roi_top_y_from_pts(roi_pts, int(h))
    elif y_top_draw is not None:
        y_top_roi = int(np.clip(int(y_top_draw), 0, int(h - 2)))
    else:
        # Best-effort fallback when ROI polygon is not provided.
        y_top_roi = int(np.clip(int(np.min(y)), 0, int(h - 2)))
    y_bot = int(h - 1)
    roi_h = max(1, int(y_bot - y_top_roi))

    # If y-values were provided in sample-index space, map them to pixel y values over the ROI span.
    if bool(y_is_index_space):
        try:
            n = int(y.size)
            if n >= 2:
                y_pix = np.linspace(float(y_top_roi), float(y_bot), num=n).round().astype(np.int32)
            else:
                y_pix = np.array([int(y_bot)], dtype=np.int32)
        except Exception:
            y_pix = y

    band_h = int(max(1, round(float(bottom_band_ratio) * float(roi_h))))
    y_band_top = int(y_bot - band_h)

    # Apply a small margin so we don't fight polygon clipping at the image border.
    margin_px = int(max(0, round(float(margin_ratio) * float(w))))

    # Convert ratios to pixel thresholds.
    min_w_px = float(max(1.0, float(min_width_ratio) * float(w)))

    # Optional max-width: if max_width_ratio is None, do not enforce a maximum.
    max_w_px: Optional[float] = None
    if max_width_ratio is not None:
        max_w_px = float(max(1.0, float(max_width_ratio) * float(w)))

    # Sanity: if provided max is smaller than min, clamp max up to min.
    if max_w_px is not None and float(max_w_px) < float(min_w_px):
        max_w_px = float(min_w_px)

    # Anchor policy: bias which boundary stays more stable when enforcing width.
    # anchor_weight in [-1, 1]:
    #   >0 favors keeping RIGHT boundary fixed (moves left more)
    #   <0 favors keeping LEFT boundary fixed (moves right more)
    aw = float(np.clip(float(anchor_weight), -1.0, 1.0))

    in_band = y_pix >= int(y_band_top)
    if not bool(np.any(in_band)):
        return xL, xR

    # Ensure ordering (left <= right) before enforcing width.
    swap = xL2 > xR2
    if np.any(swap):
        tmp = xL2.copy()
        xL2[swap] = xR2[swap]
        xR2[swap] = tmp[swap]

    mid = 0.5 * (xL2 + xR2)
    gap = xR2 - xL2

    # Enforce minimum width in the bottom band.
    too_narrow = in_band & (gap < min_w_px)
    if np.any(too_narrow):
        if aw > 0.0:
            # Keep RIGHT more stable: expand by moving LEFT boundary.
            xR_keep = xR2[too_narrow]
            xL2[too_narrow] = xR_keep - float(min_w_px)
        elif aw < 0.0:
            # Keep LEFT more stable: expand by moving RIGHT boundary.
            xL_keep = xL2[too_narrow]
            xR2[too_narrow] = xL_keep + float(min_w_px)
        else:
            # Symmetric expansion around mid.
            half = 0.5 * float(min_w_px)
            xL2[too_narrow] = mid[too_narrow] - half
            xR2[too_narrow] = mid[too_narrow] + half

    # Recompute mid/gap after any edits before applying max-width.
    mid = 0.5 * (xL2 + xR2)
    gap = xR2 - xL2

    # Optional: enforce maximum width in the bottom band.
    # Policy:
    # - If one side has clearly better pixel support in the bottom band, keep that side and move the other.
    # - If support is similar or not provided, shrink symmetrically around the midline.
    if max_w_px is not None:
        too_wide = in_band & (gap > float(max_w_px))
        if np.any(too_wide):
            # Estimate which side is "better" supported near the bottom.
            # We only use this if selection arrays were provided by the caller.
            support_L = 0
            support_R = 0
            if xsL_sel is not None and ysL_sel is not None:
                ysL_sel_i = np.asarray(ysL_sel, dtype=np.int32).reshape(-1)
                support_L = int(np.count_nonzero(ysL_sel_i >= int(y_band_top)))
            if xsR_sel is not None and ysR_sel is not None:
                ysR_sel_i = np.asarray(ysR_sel, dtype=np.int32).reshape(-1)
                support_R = int(np.count_nonzero(ysR_sel_i >= int(y_band_top)))

            # Decide whether to keep one side fixed.
            # If the supports are close, do symmetric shrink.
            # If one side is clearly stronger (>= 1.35x), keep it and move the other side only.
            keep_mode = "sym"
            if support_L > 0 or support_R > 0:
                if support_L >= int(1.35 * max(1, support_R)):
                    keep_mode = "keep_L"
                elif support_R >= int(1.35 * max(1, support_L)):
                    keep_mode = "keep_R"

            if keep_mode == "keep_L":
                # Keep left, move right inward to enforce max width.
                xR2[too_wide] = xL2[too_wide] + float(max_w_px)
            elif keep_mode == "keep_R":
                # Keep right, move left inward to enforce max width.
                xL2[too_wide] = xR2[too_wide] - float(max_w_px)
            else:
                # No assumption about a "good" side. Use anchor bias if provided; otherwise shrink symmetrically.
                if aw > 0.0:
                    # Keep RIGHT more stable: move LEFT boundary inward.
                    xR_keep = xR2[too_wide]
                    xL2[too_wide] = xR_keep - float(max_w_px)
                elif aw < 0.0:
                    # Keep LEFT more stable: move RIGHT boundary inward.
                    xL_keep = xL2[too_wide]
                    xR2[too_wide] = xL_keep + float(max_w_px)
                else:
                    half = 0.5 * float(max_w_px)
                    xL2[too_wide] = mid[too_wide] - half
                    xR2[too_wide] = mid[too_wide] + half

    # Clamp to image bounds with margin.
    xL2 = np.clip(xL2, float(margin_px), float(w - 1 - margin_px))
    xR2 = np.clip(xR2, float(margin_px), float(w - 1 - margin_px))

    # Re-ensure ordering after clamping.
    swap2 = xL2 > xR2
    if np.any(swap2):
        tmp = xL2.copy()
        xL2[swap2] = xR2[swap2]
        xR2[swap2] = tmp[swap2]

    xL_out = xL2.astype(np.float32)
    xR_out = xR2.astype(np.float32)

    # IMPORTANT: Some callers may forget to reassign the returned arrays.
    # When geometry enforcement is explicitly enabled, also try to update
    # the provided arrays in-place (if they are writable numpy arrays of the
    # same shape). This keeps production behavior unchanged because the
    # feature is OFF by default.
    try:
        if isinstance(xL_in, np.ndarray) and isinstance(xR_in, np.ndarray):
            if xL_in.shape == xL_out.shape and xR_in.shape == xR_out.shape:
                if bool(getattr(xL_in, "flags", None) is None or xL_in.flags.writeable):
                    xL_in[...] = xL_out
                if bool(getattr(xR_in, "flags", None) is None or xR_in.flags.writeable):
                    xR_in[...] = xR_out
    except Exception:
        pass

    return xL_out, xR_out

# ============================================================
#              POLY DECISION + GEOMETRY
# ============================================================

def quadratic_pair_decision_debug(
    xsL: np.ndarray,
    ysL: np.ndarray,
    xsR: np.ndarray,
    ysR: np.ndarray,
    left_fit_1: Optional[np.ndarray],
    left_fit_2: Optional[np.ndarray],
    right_fit_1: Optional[np.ndarray],
    right_fit_2: Optional[np.ndarray],
    *,
    y_top: int,
    y_bot: int,
    w: int,
    # Hybrid thresholds
    min_abs: float = 1.4,
    k_abs: float = 0.045,
    abs_cap: float = 3.0,
    rel_improve: float = 0.15,
    # Safety checks
    max_delta_px: float = 80.0,
    max_bottom_delta_px: float = 40.0,
) -> Dict[str, float]:
    """
    Compute (and expose) the exact numeric decision signals used to pick deg1 vs deg2.
    Intended for debug logging (frame folders, console, etc).
    """
    info: Dict[str, float] = {}

    if left_fit_1 is None or right_fit_1 is None:
        info.update(
            {
                "left_ok2": 0.0,
                "right_ok2": 0.0,
                "eL1": 1e18,
                "eL2": 1e18,
                "eR1": 1e18,
                "eR2": 1e18,
                "e1_sum": 1e18,
                "e2_sum": 1e18,
                "improve_sum": -1e18,
                "abs_need": 1e18,
                "rel_need": 1e18,
                "need": 1e18,
                "curve_amt": 0.0,
                "fallback_triggered": 0.0,
                "choose_deg2": 0.0,
            }
        )
        return info

    eL1 = float(robust_fit_error_abs_x(xsL, ysL, left_fit_1))
    eR1 = float(robust_fit_error_abs_x(xsR, ysR, right_fit_1))
    eL2 = float(robust_fit_error_abs_x(xsL, ysL, left_fit_2)) if left_fit_2 is not None else 1e18
    eR2 = float(robust_fit_error_abs_x(xsR, ysR, right_fit_2)) if right_fit_2 is not None else 1e18

    left_ok2 = (left_fit_2 is not None) and _shape_checks_for_deg2(
        left_fit_1,
        left_fit_2,
        y_top=int(y_top),
        y_bot=int(y_bot),
        w=int(w),
        max_delta_px=float(max_delta_px),
        max_bottom_delta_px=float(max_bottom_delta_px),
    )
    right_ok2 = (right_fit_2 is not None) and _shape_checks_for_deg2(
        right_fit_1,
        right_fit_2,
        y_top=int(y_top),
        y_bot=int(y_bot),
        w=int(w),
        max_delta_px=float(max_delta_px),
        max_bottom_delta_px=float(max_bottom_delta_px),
    )

    e1_sum = float(eL1 + eR1)
    e2_sum = float((eL2 if left_ok2 else eL1) + (eR2 if right_ok2 else eR1))
    improve_sum = float(e1_sum - e2_sum)

    abs_need = float(np.clip(float(min_abs + k_abs * e1_sum), float(min_abs), float(abs_cap)))
    rel_need = float(rel_improve) * float(max(1.0, e1_sum))
    need = float(max(abs_need, rel_need))

    choose_deg2 = (improve_sum >= need) and bool(left_ok2) and bool(right_ok2)

    curve_amt = 0.0
    fallback_triggered = 0.0

    # Mirror the curve-friendly fallback exactly (same constants, same region).
    if (not choose_deg2) and bool(left_ok2) and bool(right_ok2):
        try:
            y_vals = np.arange(int(y_top), int(y_bot) + 1, dtype=np.int32)
            if y_vals.size >= 6:
                n = int(y_vals.size)
                y_hi = y_vals[: max(3, int(0.45 * n))]

                dL = float(
                    np.max(
                        np.abs(
                            poly_eval_x_of_y(left_fit_2, y_hi, w) - poly_eval_x_of_y(left_fit_1, y_hi, w)
                        )
                    )
                )
                dR = float(
                    np.max(
                        np.abs(
                            poly_eval_x_of_y(right_fit_2, y_hi, w) - poly_eval_x_of_y(right_fit_1, y_hi, w)
                        )
                    )
                )
                curve_amt = float(max(dL, dR))
        except Exception:
            curve_amt = 0.0

        curve_trigger_px = 8.0
        allow_degrade_px = 2.0
        if (curve_amt >= curve_trigger_px) and (e2_sum <= e1_sum + allow_degrade_px):
            choose_deg2 = True
            fallback_triggered = 1.0

    info.update(
        {
            "left_ok2": 1.0 if left_ok2 else 0.0,
            "right_ok2": 1.0 if right_ok2 else 0.0,
            "eL1": eL1,
            "eL2": float(eL2),
            "eR1": eR1,
            "eR2": float(eR2),
            "e1_sum": e1_sum,
            "e2_sum": e2_sum,
            "improve_sum": improve_sum,
            "abs_need": abs_need,
            "rel_need": rel_need,
            "need": need,
            "curve_amt": float(curve_amt),
            "fallback_triggered": float(fallback_triggered),
            "choose_deg2": 1.0 if choose_deg2 else 0.0,
        }
    )
    return info

def robust_fit_error_abs_x(xs: np.ndarray, ys: np.ndarray, fit: Optional[np.ndarray]) -> float:
    """Median absolute x-error for a x(y) polynomial fit."""
    if fit is None or xs is None or ys is None or xs.size == 0:
        return 1e18
    x_hat = np.polyval(fit, ys.astype(np.float32)).astype(np.float32)
    e = np.abs(xs.astype(np.float32) - x_hat)
    return float(np.median(e)) if e.size else 1e18


def _shape_checks_for_deg2(
    fit1: np.ndarray,
    fit2: np.ndarray,
    *,
    y_top: int,
    y_bot: int,
    w: int,
    max_delta_px: float,
    max_bottom_delta_px: float,
) -> bool:
    """Return True if quadratic is 'similar enough' to linear (prevents crazy bends)."""
    if fit1 is None or fit2 is None:
        return False

    y_vals = np.arange(int(y_top), int(y_bot) + 1, dtype=np.int32)
    if y_vals.size < 3:
        return False

    x1 = poly_eval_x_of_y(fit1, y_vals, w)
    x2 = poly_eval_x_of_y(fit2, y_vals, w)
    delta = np.abs(x2 - x1)

    if float(np.max(delta)) > float(max_delta_px):
        return False

    h_seg = int(max(1, 0.30 * len(y_vals)))
    d_bot = float(np.mean(delta[-h_seg:])) if h_seg > 0 else float(np.mean(delta))
    if d_bot > float(max_bottom_delta_px):
        return False

    return True


def prefer_quadratic_pair_consistent(
    xsL: np.ndarray,
    ysL: np.ndarray,
    xsR: np.ndarray,
    ysR: np.ndarray,
    left_fit_1: Optional[np.ndarray],
    left_fit_2: Optional[np.ndarray],
    right_fit_1: Optional[np.ndarray],
    right_fit_2: Optional[np.ndarray],
    *,
    y_top: int,
    y_bot: int,
    w: int,
    # Hybrid thresholds
    min_abs: float = 1.4,
    k_abs: float = 0.045,
    abs_cap: float = 3.0,
    rel_improve: float = 0.15,
    # Safety checks
    max_delta_px: float = 80.0,
    max_bottom_delta_px: float = 40.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, int, float, float, float, float]:
    """Choose deg1 vs deg2 with a single decision for both sides.

    Returns:
      (left_fit, right_fit, left_deg, right_deg, left_e1, left_e2, right_e1, right_e2)
    """
    if left_fit_1 is None or right_fit_1 is None:
        return None, None, 0, 0, 1e18, 1e18, 1e18, 1e18

    left_e1 = robust_fit_error_abs_x(xsL, ysL, left_fit_1)
    right_e1 = robust_fit_error_abs_x(xsR, ysR, right_fit_1)

    left_e2 = robust_fit_error_abs_x(xsL, ysL, left_fit_2) if left_fit_2 is not None else 1e18
    right_e2 = robust_fit_error_abs_x(xsR, ysR, right_fit_2) if right_fit_2 is not None else 1e18

    left_ok2 = (left_fit_2 is not None) and _shape_checks_for_deg2(
        left_fit_1,
        left_fit_2,
        y_top=int(y_top),
        y_bot=int(y_bot),
        w=int(w),
        max_delta_px=float(max_delta_px),
        max_bottom_delta_px=float(max_bottom_delta_px),
    )
    right_ok2 = (right_fit_2 is not None) and _shape_checks_for_deg2(
        right_fit_1,
        right_fit_2,
        y_top=int(y_top),
        y_bot=int(y_bot),
        w=int(w),
        max_delta_px=float(max_delta_px),
        max_bottom_delta_px=float(max_bottom_delta_px),
    )

    e1_sum = float(left_e1 + right_e1)
    e2_sum = float((left_e2 if left_ok2 else left_e1) + (right_e2 if right_ok2 else right_e1))
    improve_sum = float(e1_sum - e2_sum)

    abs_need = float(np.clip(float(min_abs + k_abs * e1_sum), float(min_abs), float(abs_cap)))
    rel_need = float(rel_improve) * float(max(1.0, e1_sum))
    need = float(max(abs_need, rel_need))

    choose_deg2 = (improve_sum >= need) and left_ok2 and right_ok2

    # Curve-friendly fallback:
    # Median error is often dominated by dense bottom pixels where deg1 and deg2 are similar.
    # On curved roads, deg2 can look better higher up even if the median error doesn't strictly improve.
    if (not choose_deg2) and left_ok2 and right_ok2:
        try:
            y_vals = np.arange(int(y_top), int(y_bot) + 1, dtype=np.int32)
            if y_vals.size >= 6:
                n = int(y_vals.size)
                # Focus on upper part of the visible lane where curvature is most apparent.
                y_hi = y_vals[: max(3, int(0.45 * n))]

                dL = np.max(
                    np.abs(
                        poly_eval_x_of_y(left_fit_2, y_hi, w) - poly_eval_x_of_y(left_fit_1, y_hi, w)
                    )
                )
                dR = np.max(
                    np.abs(
                        poly_eval_x_of_y(right_fit_2, y_hi, w) - poly_eval_x_of_y(right_fit_1, y_hi, w)
                    )
                )
                curve_amt = float(max(dL, dR))
            else:
                curve_amt = 0.0
        except Exception:
            curve_amt = 0.0

        # If it's meaningfully curved and not much worse in robust error, allow deg2.
        curve_trigger_px = 8.0
        allow_degrade_px = 2.0
        if (curve_amt >= curve_trigger_px) and (e2_sum <= e1_sum + allow_degrade_px):
            choose_deg2 = True

    if choose_deg2:
        return left_fit_2, right_fit_2, 2, 2, float(left_e1), float(left_e2), float(right_e1), float(right_e2)

    return left_fit_1, right_fit_1, 1, 1, float(left_e1), float(left_e2), float(right_e1), float(right_e2)

def build_lane_polygon_from_fits(
    left_fit: np.ndarray,
    right_fit: np.ndarray,
    *,
    roi_pts: np.ndarray,
    h: int,
    w: int,
    y_top_draw: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Polygon + curves. Overlay height is controlled by y_top_draw (>= ROI top)."""
    y_top_roi = roi_top_y_from_pts(roi_pts, h)
    y_top = int(np.clip(int(y_top_draw), y_top_roi, h - 2))
    y_bot = h - 1
    if y_bot <= y_top + 10:
        return None, None, None

    y_vals = np.arange(y_top, y_bot + 1, dtype=np.int32)
    xL = poly_eval_x_of_y(left_fit, y_vals, w)
    xR = poly_eval_x_of_y(right_fit, y_vals, w)

    swap = xL > xR
    if np.any(swap):
        tmp = xL.copy()
        xL[swap] = xR[swap]
        xR[swap] = tmp[swap]

    min_gap = 40.0
    gap = xR - xL
    bad = gap < min_gap
    if np.any(bad):
        mid = 0.5 * (xL[bad] + xR[bad])
        xL[bad] = mid - 0.5 * min_gap
        xR[bad] = mid + 0.5 * min_gap

    xL = np.clip(xL, 0, w - 1)
    xR = np.clip(xR, 0, w - 1)

    left_curve = np.stack([xL.astype(np.int32), y_vals.astype(np.int32)], axis=1).reshape(-1, 1, 2)
    right_curve = np.stack([xR.astype(np.int32), y_vals.astype(np.int32)], axis=1).reshape(-1, 1, 2)

    poly_pts = (
        np.vstack([left_curve.reshape(-1, 2), right_curve.reshape(-1, 2)[::-1]])
        .astype(np.int32)
        .reshape(-1, 1, 2)
    )
    return poly_pts, left_curve, right_curve


def draw_curve_lane_overlay_from_fits(
    frame_bgr: np.ndarray,
    *,
    roi_pts: np.ndarray,
    left_fit: Optional[np.ndarray],
    right_fit: Optional[np.ndarray],
    lane_fill_alpha: float,
    overlay_top_y_ratio: float,
) -> np.ndarray:
    """Draw final lane overlay from *precomputed* polynomial fits.

    Purpose:
    - Avoid recomputing Hough/pixel-selection/fitting in the overlay step.
    - Ensure debug visuals and final overlay use the *same* fits.

    If fits are missing, returns the input frame unchanged.
    """
    out = frame_bgr.copy()
    if left_fit is None or right_fit is None:
        return out

    h, w = out.shape[:2]

    # y_top_draw controls where we START drawing the overlay (larger y -> draw lower -> less height).
    # Base policy: start at the lower of (ROI top) and (overlay ratio).
    y_top_roi = roi_top_y_from_pts(roi_pts, h)
    y_top_draw = int(max(y_top_roi, int(h * float(overlay_top_y_ratio))))

    # Push it ~1% further down to avoid upper-region clutter (e.g., vehicle blobs).
    y_top_draw = int(y_top_draw + int(0.01 * h))

    # Clamp to valid range.
    y_top_draw = int(np.clip(y_top_draw, y_top_roi, h - 2))

    poly_pts, left_curve, right_curve = build_lane_polygon_from_fits(
        left_fit,
        right_fit,
        roi_pts=roi_pts,
        h=h,
        w=w,
        y_top_draw=y_top_draw,
    )
    if poly_pts is None:
        return out

    overlay = out.copy()
    cv2.fillPoly(overlay, [poly_pts], (0, 255, 0))
    cv2.polylines(overlay, [left_curve], False, (255, 0, 255), 5, cv2.LINE_AA)
    cv2.polylines(overlay, [right_curve], False, (0, 255, 255), 5, cv2.LINE_AA)
    cv2.addWeighted(overlay, float(lane_fill_alpha), out, 1.0 - float(lane_fill_alpha), 0.0, out)

    return out

def draw_curve_lane_overlay(
    frame_bgr: np.ndarray,
    *,
    roi_bgr: np.ndarray,
    roi_pts: np.ndarray,
    best_clean: np.ndarray,
    edges_for_hough_01: np.ndarray,
    lane_fill_alpha: float,
    overlay_top_y_ratio: float,
    hough_max_dist_px: float,
    center_exclude_px: int = 70,
    force_quadratic: bool = True,
) -> np.ndarray:
    """Final overlay using Hough-proximity pixel selection.

    This is the reusable version of the debug overlay logic.
    """
    out = frame_bgr.copy()
    h, w = best_clean.shape[:2]

    y_top_roi = roi_top_y_from_pts(roi_pts, h)

    # Start drawing at the lower of (ROI top) and (overlay ratio), then push further down.
    y_top_draw = int(max(y_top_roi, int(h * float(overlay_top_y_ratio))))
    y_top_draw = int(y_top_draw + int(0.01 * h))
    y_top_draw = int(np.clip(y_top_draw, y_top_roi, h - 2))

    # --- Option 1 (improved): remove center blobs via connected components
    # We do this BOTH on the mask used for pixel selection AND on the edge image used for Hough.

    # Work on local copies (do not mutate caller's arrays)
    best_for_fit = suppress_center_blob_components(best_clean)

    edges_for_hough_use = edges_for_hough_01.copy() if edges_for_hough_01 is not None else None
    if edges_for_hough_use is not None:
        # First: suppress big center-ish edge components directly (vehicle edges tend to dominate Hough)
        edges_for_hough_use = suppress_center_blob_edges(edges_for_hough_use)

        # Also remove edges in regions where the blob was removed from the mask (mask-based cue)
        removed = ((best_clean > 0) & (best_for_fit == 0)).astype(np.uint8) * 255
        if int(cv2.countNonZero(removed)) > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
            removed = cv2.dilate(removed, k, iterations=1)
            edges_for_hough_use[removed > 0] = 0

        # Keep the old narrow center band as a final safety-net
        cx = int(max(0, int(center_exclude_px)))
        if cx > 0:
            mid = int(w // 2)
            x0 = int(max(0, mid - cx))
            x1 = int(min(w, mid + cx))
            y_cut = int(np.clip(int(0.92 * h), 0, h))
            edges_for_hough_use[:y_cut, x0:x1] = 0

    # Recompute Hough lines on the (possibly) suppressed edge image
    lines = hough_lines_p(edges_for_hough_use)
    left_lines, right_lines = split_hough_lines_left_right(lines, w=w)

    # Select lane pixels from the (possibly) suppressed mask
    xsL, ysL, xsR, ysR = select_lane_pixels_by_hough_proximity(
        best_for_fit,
        left_lines,
        right_lines,
        max_dist_px=float(hough_max_dist_px),
    )

    haveL = len(left_lines) > 0
    haveR = len(right_lines) > 0

    # Side-specific salvage: sometimes one side has Hough lines but loses pixels due to cross-side competition.
    # If that happens, re-select for that side only using a larger distance and no L/R comparison.
    if haveL and xsL.size < 80:
        ys_all, xs_all = np.where(best_for_fit > 0)
        if xs_all.size > 0:
            px, py = xs_all.astype(np.float32), ys_all.astype(np.float32)
            big = 1e9
            dL_only = np.full(px.shape, big, dtype=np.float32)
            for x1, y1, x2, y2, _ in left_lines:
                # reuse the same point-to-seg distance logic via an inner helper
                vx, vy = x2 - x1, y2 - y1
                denom = vx * vx + vy * vy
                if denom < 1e-6:
                    d = np.hypot(px - x1, py - y1)
                else:
                    t = np.clip(((px - x1) * vx + (py - y1) * vy) / denom, 0.0, 1.0)
                    d = np.hypot(px - (x1 + t * vx), py - (y1 + t * vy))
                dL_only = np.minimum(dL_only, d)
            keep = dL_only <= float(hough_max_dist_px) * 3.0
            if int(np.count_nonzero(keep)) >= 120:
                xsL, ysL = xs_all[keep].astype(np.int32), ys_all[keep].astype(np.int32)

    if haveR and xsR.size < 80:
        ys_all, xs_all = np.where(best_for_fit > 0)
        if xs_all.size > 0:
            px, py = xs_all.astype(np.float32), ys_all.astype(np.float32)
            big = 1e9
            dR_only = np.full(px.shape, big, dtype=np.float32)
            for x1, y1, x2, y2, _ in right_lines:
                vx, vy = x2 - x1, y2 - y1
                denom = vx * vx + vy * vy
                if denom < 1e-6:
                    d = np.hypot(px - x1, py - y1)
                else:
                    t = np.clip(((px - x1) * vx + (py - y1) * vy) / denom, 0.0, 1.0)
                    d = np.hypot(px - (x1 + t * vx), py - (y1 + t * vy))
                dR_only = np.minimum(dR_only, d)
            keep = dR_only <= float(hough_max_dist_px) * 3.0
            if int(np.count_nonzero(keep)) >= 120:
                xsR, ysR = xs_all[keep].astype(np.int32), ys_all[keep].astype(np.int32)

    # Same fallbacks as before, but using best_for_fit (center-suppressed mask)
    if (xsL.size < 120 and haveL) or (xsR.size < 120 and haveR):
        xsL2, ysL2, xsR2, ysR2 = select_lane_pixels_by_hough_proximity(
            best_for_fit,
            left_lines,
            right_lines,
            max_dist_px=float(hough_max_dist_px) * 1.8,
        )
        if xsL.size < 120 and haveL and xsL2.size > xsL.size:
            xsL, ysL = xsL2, ysL2
        if xsR.size < 120 and haveR and xsR2.size > xsR.size:
            xsR, ysR = xsR2, ysR2

    if (xsL.size < 120 and not haveL) or (xsR.size < 120 and not haveR):
        ys_all, xs_all = np.where(best_for_fit > 0)
        if xs_all.size >= 300:
            mid2 = int(w // 2)
            li = xs_all < mid2
            ri = ~li
            if xsL.size < 120 and (not haveL) and np.count_nonzero(li) > 150:
                xsL, ysL = xs_all[li].astype(np.int32), ys_all[li].astype(np.int32)
            if xsR.size < 120 and (not haveR) and np.count_nonzero(ri) > 150:
                xsR, ysR = xs_all[ri].astype(np.int32), ys_all[ri].astype(np.int32)

    left_fit_1 = fit_poly_x_of_y(xsL, ysL, degree=1)
    left_fit_2 = fit_poly_x_of_y(xsL, ysL, degree=2)
    right_fit_1 = fit_poly_x_of_y(xsR, ysR, degree=1)
    right_fit_2 = fit_poly_x_of_y(xsR, ysR, degree=2)

    y_top = y_top_roi
    y_bot = h - 1

    if bool(force_quadratic):
        left_fit = left_fit_2 if left_fit_2 is not None else left_fit_1
        right_fit = right_fit_2 if right_fit_2 is not None else right_fit_1
    else:
        left_fit, right_fit, _left_deg, _right_deg, _le1, _le2, _re1, _re2 = prefer_quadratic_pair_consistent(
            xsL,
            ysL,
            xsR,
            ysR,
            left_fit_1,
            left_fit_2,
            right_fit_1,
            right_fit_2,
            y_top=y_top,
            y_bot=y_bot,
            w=w,
        )

    if left_fit is None or right_fit is None:
        return out

    poly_pts, left_curve, right_curve = build_lane_polygon_from_fits(
        left_fit,
        right_fit,
        roi_pts=roi_pts,
        h=h,
        w=w,
        y_top_draw=y_top_draw,
    )
    if poly_pts is None:
        return out

    overlay = out.copy()
    cv2.fillPoly(overlay, [poly_pts], (0, 255, 0))
    cv2.polylines(overlay, [left_curve], False, (255, 0, 255), 5, cv2.LINE_AA)
    cv2.polylines(overlay, [right_curve], False, (0, 255, 255), 5, cv2.LINE_AA)
    cv2.addWeighted(overlay, float(lane_fill_alpha), out, 1.0 - float(lane_fill_alpha), 0.0, out)

    return out




# ============================================================
#              MASK SCORING
# ============================================================

def mask_stats(mask_01: np.ndarray):
    h, w = mask_01.shape[:2]
    area = float(h * w) if h and w else 1.0
    nz = int(cv2.countNonZero(mask_01))
    cov = float(nz) / area

    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask_01 > 0).astype(np.uint8), connectivity=8)
    cc = max(0, int(num) - 1)

    largest = top2 = topk = 0
    elong = 0.0
    k = 8

    if cc > 0:
        areas = stats[1:, 4]
        if len(areas):
            areas_sorted = np.sort(areas)[::-1]
            largest = int(areas_sorted[0])
            top2 = int(areas_sorted[0] + (areas_sorted[1] if len(areas_sorted) > 1 else 0))
            topk = int(np.sum(areas_sorted[: min(k, len(areas_sorted))]))

            idx_largest = 1 + int(np.argmax(areas))
            _, _, bw, bh, _a = stats[idx_largest]
            short = float(max(1, min(bw, bh)))
            long = float(max(bw, bh))
            elong = long / short

    denom = float(max(1, nz))
    return (
        cov,
        cc,
        float(largest) / denom if nz else 0.0,
        float(top2) / denom if nz else 0.0,
        float(topk) / denom if nz else 0.0,
        elong,
    )


def bottom_lr_balance(mask_01: np.ndarray, bottom_ratio: float = 0.35):
    h, w = mask_01.shape[:2]
    y0 = int(max(0, h - int(h * bottom_ratio)))
    band = mask_01[y0:h, :]
    nz = int(cv2.countNonZero(band))
    if nz == 0:
        return 0.0, 0.0, 0.0

    mid = w // 2
    lf = float(cv2.countNonZero(band[:, :mid])) / nz
    rf = float(cv2.countNonZero(band[:, mid:])) / nz
    bal = 1.0 - abs(lf - rf)
    return lf, rf, float(np.clip(bal, 0.0, 1.0))


def score_mask(mask_01: np.ndarray, ignore_top_ratio: float):
    mm = mask_01.copy()
    h, w = mm.shape[:2]
    y_cut = int(h * ignore_top_ratio)
    mm[:y_cut, :] = 0

    cov, cc, largest_frac, _t2, topk_frac, elong = mask_stats(mm)
    lf, rf, bal = bottom_lr_balance(mm)

    if cov < 0.0004 or cov > 0.35:
        return -1e9

    lane_reward = 2.6 * topk_frac * min(1.0, cov / 0.012)
    lr_reward = 0.9 * (1.0 if (lf > 0.18 and rf > 0.18) else 0.0) + 0.5 * bal
    blob_pen = 1.4 * max(0.0, largest_frac - 0.85)
    cc_pen = float(cc) / 900.0
    shape_reward = 0.12 * min(float(elong), 20.0)

    return lane_reward + lr_reward + shape_reward - blob_pen - cc_pen


def pick_best_scored(masks, ignore_top_ratio: float):
    best_name, best_mask = masks[0]
    best_score = -1e18
    scores = {}
    for name, m in masks:
        s = float(score_mask(m, ignore_top_ratio))
        scores[name] = s
        if s > best_score or (abs(s - best_score) <= 0.12 and name == "GRAD"):
            best_name, best_mask, best_score = name, m, s
    return best_name, best_mask, best_score, scores


# ============================================================
#              HOUGH HELPERS
# ============================================================

def hough_lines_p(edges_01, rho=1.0, theta_deg=1.0, threshold=10, min_line_len=10, max_line_gap=110):
    if edges_01 is None:
        return None
    return cv2.HoughLinesP(
        edges_01,
        rho=float(rho),
        theta=np.deg2rad(float(theta_deg)),
        threshold=int(threshold),
        minLineLength=int(min_line_len),
        maxLineGap=int(max_line_gap),
    )


def split_hough_lines_left_right(lines, w, min_abs_slope=0.25):
    left, right = [], []
    if lines is None:
        return left, right
    mid = w / 2.0
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx, dy = x2 - x1, y2 - y1
        if abs(dx) < 1e-6:
            continue
        slope = dy / dx
        if abs(slope) < min_abs_slope:
            continue
        x_mid = 0.5 * (x1 + x2)
        if slope < 0 and x_mid < mid:
            left.append((x1, y1, x2, y2, slope))
        elif slope > 0 and x_mid >= mid:
            right.append((x1, y1, x2, y2, slope))
    return left, right


def select_lane_pixels_by_hough_proximity(mask_clean_01, left_lines, right_lines, max_dist_px=10.0):
    ys, xs = np.where(mask_clean_01 > 0)
    if xs.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    def dist_point_seg(px, py, x1, y1, x2, y2):
        vx, vy = x2 - x1, y2 - y1
        denom = vx * vx + vy * vy
        if denom < 1e-6:
            return np.hypot(px - x1, py - y1)
        t = np.clip(((px - x1) * vx + (py - y1) * vy) / denom, 0.0, 1.0)
        return np.hypot(px - (x1 + t * vx), py - (y1 + t * vy))

    px, py = xs.astype(np.float32), ys.astype(np.float32)
    big = 1e9
    dL = np.full(px.shape, big, dtype=np.float32)
    dR = np.full(px.shape, big, dtype=np.float32)

    for x1, y1, x2, y2, _ in left_lines:
        dL = np.minimum(dL, dist_point_seg(px, py, x1, y1, x2, y2))
    for x1, y1, x2, y2, _ in right_lines:
        dR = np.minimum(dR, dist_point_seg(px, py, x1, y1, x2, y2))

    keepL = (dL <= max_dist_px) & (dL <= dR + 1.0)
    keepR = (dR <= max_dist_px) & (dR < dL + 1.0)
    return xs[keepL], ys[keepL], xs[keepR], ys[keepR]


# ============================================================
#              POLYNOMIAL FITTING
# ============================================================

def fit_poly_x_of_y(xs, ys, degree=2):
    if xs is None or ys is None or xs.size < max(80, degree + 1):
        return None
    if xs.size > 7000:
        idx = np.random.choice(xs.size, size=7000, replace=False)
        xs, ys = xs[idx], ys[idx]
    try:
        return np.polyfit(ys.astype(np.float32), xs.astype(np.float32), int(degree))
    except Exception:
        return None


def poly_eval_x_of_y(coeffs, ys, w):
    return np.clip(np.polyval(coeffs, ys.astype(np.float32)), 0, w - 1)




# ============================================================
#              PUBLIC EXPORTS
# ============================================================

__all__ = [
    # Small utilities
    "to_mask_01",

    # Defaults / config
    "CurveDefaults",
    "get_curve_defaults",
    "resolve_curve_params",

    # ROI builders (curve runner + debug)
    "build_alt_rois_from_main_roi",

    # Legacy shared helpers
    "apply_roi_mask",
    "apply_color_threshold",
    "apply_canny",
    "detect_lines_hough",

    # Masks / cleanup
    "mask_hsv_curve_tuned",
    "mask_hls_clahe",
    "mask_gradient_edge",
    "filter_grad_by_slope",
    "clean_mask_light",
    "suppress_center_blob_components",
    "suppress_center_blob_edges",

    # Merging / gating
    "merge_grad_with_color_for_score",
    "build_corridor_from_mask",
    "merge_masks_for_hough_union",
    "edges_for_hough",

    # ROI helpers
    "roi_mask_full_frame",
    "alt_roi_rect_plus_trapezoid",
    "roi_top_y_from_pts",

    # Fit + geometry decisions
    "robust_fit_error_abs_x",
    "quadratic_pair_decision_debug",
    "prefer_quadratic_pair_consistent",
    "enforce_min_lane_width_bottom_band",
    "fit_poly_x_of_y",
    "poly_eval_x_of_y",

    # Hough helpers
    "hough_lines_p",
    "split_hough_lines_left_right",
    "select_lane_pixels_by_hough_proximity",

    # Overlay
    "build_lane_polygon_from_fits",
    "draw_curve_lane_overlay_from_fits",
    "draw_curve_lane_overlay",

    # Mask scoring
    "mask_stats",
    "bottom_lr_balance",
    "score_mask",
    "pick_best_scored",
]