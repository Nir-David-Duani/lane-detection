"""
Curve Detection - Standalone Video Runner
==========================================
This module processes driving videos and detects left and right lane boundaries on curves.

The pipeline includes:
1. ROI masking with asymmetric left/right regions
2. Color masking (HSV and HLS with CLAHE)
3. Edge detection and Hough line detection
4. Polynomial fitting (linear and quadratic)
5. Temporal smoothing using history and EMA
6. Lane change detection based on center shift
7. Lane polygon overlay visualization

Key features:
- Standalone runnable module
- Temporal smoothing: history + EMA (smoothing x(y) samples then refit)
- Lane change detection using center-shift heuristic
- Geometry enforcement options for lane width constraints

Run examples:
  python3 -m src.enhancements.curve_detection --video data/processed/clip_20251223_015741_video.mp4 --output output/curve_out.mp4
  python3 src/enhancements/curve_detection.py --video data/processed/clip_20251223_015741_video.mp4 --output output/curve_out.mp4

Notes:
- This runner relies on curve_detection_pipeline (single source of truth) - no local fallbacks.
"""

import os
import sys
import warnings
from collections import deque
from typing import Dict, List, Optional, Tuple

import argparse
import cv2
import numpy as np
from numpy.polynomial.polyutils import RankWarning

# -----------------------------------------------------------------------------
# Path setup - keep this module runnable both as:
#   python3 -m src.enhancements.curve_detection
# and as:
#   python3 src/enhancements/curve_detection.py
#
# Layout:
#   <project_root>/data/
#   <project_root>/lane-detection/src/...
# -----------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory layout:
#   <repo_root>/
#     data/
#     lane-detection/
#       src/
#         enhancements/
#
# This file lives at: <repo_root>/lane-detection/src/enhancements/curve_detection.py

# lane-detection/src
_SRC_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
# lane-detection
_LANE_DETECTION_ROOT = os.path.abspath(os.path.join(_SRC_DIR, ".."))
# repo_root (contains data/)
_PROJECT_ROOT = os.path.abspath(os.path.join(_LANE_DETECTION_ROOT, ".."))

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# IMPORTANT: this runner must use ONLY curve_detection_pipeline (no pipeline.py).
import curve_detection_pipeline as cdp  # noqa: E402


warnings.simplefilter("ignore", RankWarning)

# ============================================================
#              CURVE DETECTION PIPELINE IMPORTS
# ============================================================
# Curve detection uses ONLY `curve_detection_pipeline` (cdp).
# No dependency on `pipeline.py`.

# Short aliases to the building blocks we rely on.
mask_hsv_curve_tuned = cdp.mask_hsv_curve_tuned
mask_hls_clahe = cdp.mask_hls_clahe
clean_mask_light = cdp.clean_mask_light
build_corridor_from_mask = cdp.build_corridor_from_mask
edges_for_hough = cdp.edges_for_hough
hough_lines_p = cdp.hough_lines_p
split_hough_lines_left_right = cdp.split_hough_lines_left_right
select_lane_pixels_by_hough_proximity = cdp.select_lane_pixels_by_hough_proximity
fit_poly_x_of_y = cdp.fit_poly_x_of_y
build_lane_polygon_from_fits = cdp.build_lane_polygon_from_fits

roi_top_y_from_pts = cdp.roi_top_y_from_pts

# ============================================================
#              PARAMETER DEFAULTS
# ============================================================
# Single source of truth lives in curve_detection_pipeline

def _get_curve_params_from_pipeline(overrides: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    ov = dict(overrides or {})

    def _to_dict(obj: object) -> Dict[str, object]:
        # Accept dict-like
        if isinstance(obj, dict):
            return dict(obj)

        # Dataclasses (preferred)
        try:
            import dataclasses

            if dataclasses.is_dataclass(obj):
                return dict(dataclasses.asdict(obj))  # type: ignore[arg-type]
        except Exception:
            pass

        # Plain objects with attributes
        if hasattr(obj, "__dict__"):
            try:
                return dict(getattr(obj, "__dict__"))
            except Exception:
                pass

        # Mapping-like
        try:
            return dict(obj)  # type: ignore[arg-type]
        except Exception as e:
            raise TypeError(
                f"resolve_curve_params/get_curve_defaults returned unsupported type: {type(obj).__name__}"
            ) from e

    if hasattr(cdp, "resolve_curve_params"):
        resolved = cdp.resolve_curve_params(overrides=ov)  # type: ignore[attr-defined]
        return _to_dict(resolved)

    if hasattr(cdp, "get_curve_defaults"):
        resolved = cdp.get_curve_defaults()  # type: ignore[attr-defined]
        base = _to_dict(resolved)
        base.update(ov)
        return base

    raise RuntimeError(
        "curve_detection_pipeline must expose resolve_curve_params(overrides=...) or get_curve_defaults(). "
        "No local defaults/fallbacks are allowed in curve_detection.py."
    )


def _build_alt_rois(
    frame_bgr: np.ndarray,
    *,
    roi_pts: np.ndarray,
    top_y_ratio: float,
    left_bottom_ratio: float,
    right_bottom_ratio: float,
    top_left_x_ratio: float,
    top_right_x_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build asymmetric ALT L/R ROI-masked frames via curve_detection_pipeline.

    No fallbacks here - if the helper is missing, fail loudly.
    """
    if not hasattr(cdp, "build_alt_rois_from_main_roi"):
        raise RuntimeError(
            "curve_detection_pipeline must expose build_alt_rois_from_main_roi(...). "
            "No local ALT-ROI fallback is allowed in curve_detection.py."
        )

    out = cdp.build_alt_rois_from_main_roi(  # type: ignore[attr-defined]
        frame_bgr,
        roi_pts=roi_pts,
        top_y_ratio=float(top_y_ratio),
        left_bottom_ratio=float(left_bottom_ratio),
        right_bottom_ratio=float(right_bottom_ratio),
        top_left_x_ratio=float(top_left_x_ratio),
        top_right_x_ratio=float(top_right_x_ratio),
    )
    if isinstance(out, dict):
        return out["roi_alt_L_bgr"], out["roi_alt_R_bgr"]
    return out[0], out[1]


def apply_roi_mask(
    frame_bgr: np.ndarray,
    *,
    top_y_ratio: float,
    left_bottom_ratio: float,
    right_bottom_ratio: float,
    top_left_x_ratio: float,
    top_right_x_ratio: float,
    bottom_y_ratio: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply trapezoid ROI mask via curve_detection_pipeline.

    No local ROI fallback is allowed here.
    """
    if not hasattr(cdp, "apply_roi_mask"):
        raise RuntimeError(
            "curve_detection_pipeline must expose apply_roi_mask(...). "
            "No local ROI fallback is allowed in curve_detection.py."
        )
    return cdp.apply_roi_mask(  # type: ignore[attr-defined]
        frame_bgr,
        top_y_ratio=float(top_y_ratio),
        left_bottom_ratio=float(left_bottom_ratio),
        right_bottom_ratio=float(right_bottom_ratio),
        top_left_x_ratio=float(top_left_x_ratio),
        top_right_x_ratio=float(top_right_x_ratio),
        bottom_y_ratio=float(bottom_y_ratio),
    )



# ============================================================
#              UI HELPERS
# ============================================================

def _put_label(
    img_bgr: np.ndarray,
    text: str,
    *,
    xy: Tuple[int, int] = (20, 50),
    scale: float = 1.0,
    thickness: int = 2,
) -> None:
    """
    Draw readable white text with black outline.
    
    Args:
        img_bgr: BGR image to draw on
        text: Text string to draw
        xy: Position tuple (x, y) for text (default: (20, 50))
        scale: Font scale (default: 1.0)
        thickness: Text thickness (default: 2)
    
    Returns:
        None: Image is modified in-place
    """
    x, y = int(xy[0]), int(xy[1])
    cv2.putText(img_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, float(scale), (0, 0, 0), int(thickness) + 3, cv2.LINE_AA)
    cv2.putText(img_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, float(scale), (255, 255, 255), int(thickness), cv2.LINE_AA)


def _put_center_alert_red(img_bgr: np.ndarray, text: str, *, y: int, scale: float, thickness: int = 4) -> None:
    """
    Draw big centered red text with a black outline.
    
    Args:
        img_bgr: BGR image to draw on
        text: Text string to draw
        y: Vertical position for text
        scale: Font scale
        thickness: Text thickness (default: 4)
    
    Returns:
        None: Image is modified in-place
    """
    h, w = img_bgr.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, float(scale), int(thickness))
    x = int(max(0, (w - tw) // 2))
    y = int(np.clip(y, th + 5, h - 5))
    cv2.putText(img_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, float(scale), (0, 0, 0), int(thickness) + 4, cv2.LINE_AA)
    cv2.putText(img_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, float(scale), (0, 0, 255), int(thickness), cv2.LINE_AA)


def _render_lane_change_screen(img_bgr: np.ndarray, *, direction: str) -> None:
    """
    Overlay a lane-change warning screen (no lane overlay).
    
    Args:
        img_bgr: BGR image to draw on
        direction: Direction string ("Left" or "Right")
    
    Returns:
        None: Image is modified in-place
    """
    h, _w = img_bgr.shape[:2]
    y1 = int(0.48 * h)
    y2 = int(0.58 * h)
    _put_center_alert_red(img_bgr, "LANE CHANGE DETECTED", y=y1, scale=1.6, thickness=5)
    _put_center_alert_red(img_bgr, f"Direction: {direction}", y=y2, scale=1.0, thickness=4)

# ============================================================
#              LANE-CHANGE DETECTION
# ============================================================
# Lane-change detection from Hough lines (robust, multi-height)

def _hough_center_estimate(
    left_lines: List[Tuple[int, int, int, int, float]],
    right_lines: List[Tuple[int, int, int, int, float]],
    *,
    w: int,
    h: int,
    y_frac: float = 0.90,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Estimate lane center/width from Hough lines at a given y fraction.
    
    Args:
        left_lines: List of left lane line segments with slope
        right_lines: List of right lane line segments with slope
        w: Frame width
        h: Frame height
        y_frac: Y fraction (0-1) to evaluate at (default: 0.90)
    
    Returns:
        tuple: (center_x, width) or (None, None) if estimation fails
    """
    if w <= 0 or h <= 0:
        return None, None

    y = float(np.clip(float(y_frac) * float(h - 1), 0.0, float(h - 1)))

    def _x_at_y(line: Tuple[int, int, int, int, float]) -> Optional[float]:
        x1, y1, x2, y2, _s = line
        x1f, y1f, x2f, y2f = float(x1), float(y1), float(x2), float(y2)
        dy = (y2f - y1f)
        dx = (x2f - x1f)
        if abs(dy) < 1e-6:
            return None
        t = (y - y1f) / dy
        x = x1f + t * dx
        if not np.isfinite(x):
            return None
        return float(np.clip(x, 0.0, float(w - 1)))

    left_xs: List[float] = []
    right_xs: List[float] = []

    for ln in left_lines:
        xv = _x_at_y(ln)
        if xv is not None:
            left_xs.append(float(xv))

    for ln in right_lines:
        xv = _x_at_y(ln)
        if xv is not None:
            right_xs.append(float(xv))

    if len(left_xs) == 0 or len(right_xs) == 0:
        return None, None

    xL = float(np.median(np.array(left_xs, dtype=np.float64)))
    xR = float(np.median(np.array(right_xs, dtype=np.float64)))
    if not (np.isfinite(xL) and np.isfinite(xR)):
        return None, None
    if xR <= xL:
        return None, None

    return float(0.5 * (xL + xR)), float(xR - xL)


def _hough_center_estimate_multi(
    left_lines: List[Tuple[int, int, int, int, float]],
    right_lines: List[Tuple[int, int, int, int, float]],
    *,
    w: int,
    h: int,
    y_fracs: List[float],
) -> Tuple[Optional[float], Optional[float], int]:
    """
    Median center/width across multiple y heights.
    
    Args:
        left_lines: List of left lane line segments with slope
        right_lines: List of right lane line segments with slope
        w: Frame width
        h: Frame height
        y_fracs: List of y fractions (0-1) to evaluate at
    
    Returns:
        tuple: (center_x, width, num_valid) or (None, None, 0) if estimation fails
    """
    centers: List[float] = []
    widths: List[float] = []
    for f in y_fracs:
        c, wd = _hough_center_estimate(left_lines, right_lines, w=w, h=h, y_frac=float(f))
        if c is None or wd is None:
            continue
        if not (np.isfinite(c) and np.isfinite(wd)):
            continue
        centers.append(float(c))
        widths.append(float(wd))

    if len(centers) == 0:
        return None, None, 0

    center_med = float(np.median(np.array(centers, dtype=np.float64)))
    width_med = float(np.median(np.array(widths, dtype=np.float64)))
    return center_med, width_med, int(len(centers))

# ============================================================
#              MAIN PROCESSING
# ============================================================

def process_video_with_curve_prediction(
    *,
    video_path: str,
    output_path: str,
    top_y_ratio: Optional[float] = None,
    left_bottom_ratio: Optional[float] = None,
    right_bottom_ratio: Optional[float] = None,
    top_left_x_ratio: Optional[float] = None,
    top_right_x_ratio: Optional[float] = None,
    overlay_top_y_ratio: Optional[float] = None,
    lane_fill_alpha: Optional[float] = None,
    clahe_clip: Optional[float] = None,
    clahe_grid: Optional[int] = None,
    merge_grad_dist_px: Optional[int] = None,
    hough_dist_px: Optional[float] = None,
    geom_enforce_bottom_width: bool = False,
    geom_min_width_ratio: float = 0.22,
    geom_max_width_ratio: Optional[float] = None,
    geom_bottom_band_ratio: float = 0.30,
    geom_margin_ratio: float = 0.02,
    geom_anchor_weight: float = 0.0,
    geom_smooth_alpha: float = 0.18,
) -> None:
    """
    Run curve detection on a video and write an annotated output video.
    
    Args:
        video_path: Path to input video file
        output_path: Path to save output video
        top_y_ratio: ROI top Y ratio (0-1, optional)
        left_bottom_ratio: ROI left bottom ratio (0-1, optional)
        right_bottom_ratio: ROI right bottom ratio (0-1, optional)
        top_left_x_ratio: ROI top left X ratio (0-1, optional)
        top_right_x_ratio: ROI top right X ratio (0-1, optional)
        overlay_top_y_ratio: Overlay top Y ratio (0-1, optional)
        lane_fill_alpha: Lane fill transparency (0-1, optional)
        clahe_clip: CLAHE clip limit (optional)
        clahe_grid: CLAHE grid size (optional)
        merge_grad_dist_px: Merge gradient distance in pixels (optional)
        hough_dist_px: Hough proximity distance in pixels (optional)
        geom_enforce_bottom_width: Enable geometry enforcement (default: False)
        geom_min_width_ratio: Minimum lane width ratio (default: 0.22)
        geom_max_width_ratio: Maximum lane width ratio (optional)
        geom_bottom_band_ratio: Bottom band ratio for enforcement (default: 0.30)
        geom_margin_ratio: Margin ratio for enforcement (default: 0.02)
        geom_anchor_weight: Anchor weight for enforcement (default: 0.0)
        geom_smooth_alpha: Smoothing alpha for enforcement (default: 0.18)
    
    Returns:
        None: Output video is saved to output_path
    """
    overrides: Dict[str, object] = {}
    if top_y_ratio is not None:
        overrides["top_y_ratio"] = float(top_y_ratio)
    if left_bottom_ratio is not None:
        overrides["left_bottom_ratio"] = float(left_bottom_ratio)
    if right_bottom_ratio is not None:
        overrides["right_bottom_ratio"] = float(right_bottom_ratio)
    if top_left_x_ratio is not None:
        overrides["top_left_x_ratio"] = float(top_left_x_ratio)
    if top_right_x_ratio is not None:
        overrides["top_right_x_ratio"] = float(top_right_x_ratio)
    if overlay_top_y_ratio is not None:
        overrides["overlay_top_y_ratio"] = float(overlay_top_y_ratio)
    if lane_fill_alpha is not None:
        overrides["lane_fill_alpha"] = float(lane_fill_alpha)
    if clahe_clip is not None:
        overrides["clahe_clip"] = float(clahe_clip)
    if clahe_grid is not None:
        overrides["clahe_grid"] = int(clahe_grid)
    if merge_grad_dist_px is not None:
        overrides["merge_grad_dist_px"] = int(merge_grad_dist_px)
    if hough_dist_px is not None:
        overrides["hough_dist_px"] = float(hough_dist_px)

    params = _get_curve_params_from_pipeline(overrides=overrides)

    # Local typed aliases for readability
    top_y_ratio_f = float(params["top_y_ratio"])
    left_bottom_ratio_f = float(params["left_bottom_ratio"])
    right_bottom_ratio_f = float(params["right_bottom_ratio"])
    top_left_x_ratio_f = float(params["top_left_x_ratio"])
    top_right_x_ratio_f = float(params["top_right_x_ratio"])
    overlay_top_y_ratio_f = float(params["overlay_top_y_ratio"])
    lane_fill_alpha_f = float(params["lane_fill_alpha"])
    clahe_clip_f = float(params["clahe_clip"])
    clahe_grid_i = int(params["clahe_grid"])
    merge_grad_dist_px_i = int(params["merge_grad_dist_px"])
    hough_dist_px_f = float(params["hough_dist_px"])
    video_abs = os.path.abspath(video_path)
    if not os.path.exists(video_abs):
        raise FileNotFoundError(f"Input video not found: {video_abs}")

    cap = cv2.VideoCapture(video_abs)
    if not cap.isOpened():
        raise RuntimeError(
            "OpenCV could not open the input video (file exists, but decode/open failed). "
            f"Input: {video_abs}\n"
            "Try re-encoding (example):\n"
            f"  ffmpeg -y -i \"{video_abs}\" -c:v libx264 -pix_fmt yuv420p -movflags +faststart \"{os.path.splitext(video_abs)[0]}_reencoded.mp4\""
        )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if w <= 0 or h <= 0:
        ok, fr = cap.read()
        if not ok or fr is None:
            raise RuntimeError("Could not read first frame to infer size")
        h, w = fr.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(os.path.abspath(output_path), fourcc, float(fps), (int(w), int(h)))

    # Pixel-to-meter scale for curvature estimates (typical values).
    xm_per_pix = 3.7 / 700.0
    ym_per_pix = 30.0 / 720.0

    prev_width: Optional[float] = None

    # Persistent smoothing for geometry-enforced bottom-band xL/xR
    smooth_geom_left: Optional[np.ndarray] = None
    smooth_geom_right: Optional[np.ndarray] = None

    # Smoothing state: smooth x(y) samples then refit (more visually stable than smoothing coeffs).
    ema_alpha = 0.08
    ema_alpha_bottom = max(0.005, min(0.99, ema_alpha * 0.30))
    ema_alpha_top = max(0.010, min(0.99, ema_alpha))
    ema_alpha_very_bottom = max(0.001, min(0.99, ema_alpha * 0.20))

    history_x_size = 20
    hist_left_x: deque = deque(maxlen=int(max(1, history_x_size)))
    hist_right_x: deque = deque(maxlen=int(max(1, history_x_size)))
    smooth_left_x: Optional[np.ndarray] = None
    smooth_right_x: Optional[np.ndarray] = None

    # Lane-change warning cooldown (~2.5 seconds).
    lane_change_cooldown_frames = int(max(1, round(2.5 * float(fps))))
    lane_change_cooldown = 0
    lane_change_last_dir = "Unknown"

    # Debounce: require lane-change condition to persist across consecutive frames.
    lane_change_streak = 0
    lane_change_required_streak = 3

    # Baseline-vs-current detector (captures gradual drift).
    baseline_center_window = 12
    baseline_center_min_valid = 6
    baseline_centers: deque = deque(maxlen=int(max(1, baseline_center_window)))

    def _ema_update(prev: Optional[np.ndarray], cur: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Piecewise EMA: stronger smoothing near bottom, lighter near top."""
        if cur is None:
            return prev
        c = np.asarray(cur, dtype=np.float64).reshape(-1)
        if prev is None:
            return c
        p = np.asarray(prev, dtype=np.float64).reshape(-1)
        if p.shape != c.shape:
            return c

        n = int(c.size)
        if n <= 1:
            return (1.0 - ema_alpha) * p + ema_alpha * c

        out = p.copy()
        i_top = int(round(0.62 * n))
        i_mid = int(round(0.85 * n))

        a_top = float(ema_alpha_top)
        a_mid = float(ema_alpha_bottom)
        a_bot = float(ema_alpha_very_bottom)

        if i_top > 0:
            out[:i_top] = (1.0 - a_top) * p[:i_top] + a_top * c[:i_top]
        if i_mid > i_top:
            out[i_top:i_mid] = (1.0 - a_mid) * p[i_top:i_mid] + a_mid * c[i_top:i_mid]
        if i_mid < n:
            out[i_mid:] = (1.0 - a_bot) * p[i_mid:] + a_bot * c[i_mid:]

        return out

    def _fit_from_xy(ys_i: np.ndarray, xs_f: np.ndarray, degree: int = 2) -> Optional[np.ndarray]:
        if ys_i is None or xs_f is None:
            return None
        if ys_i.size < max(20, degree + 1):
            return None
        try:
            return np.polyfit(ys_i.astype(np.float32), xs_f.astype(np.float32), int(degree))
        except Exception:
            return None

    def _median_from_hist(hist: deque) -> Optional[np.ndarray]:
        arrs = [np.asarray(a, dtype=np.float64).reshape(-1) for a in hist if a is not None]
        if len(arrs) == 0:
            return None
        try:
            stack = np.stack(arrs, axis=0)  # (T, K)
            return np.median(stack, axis=0)
        except Exception:
            return None

    progress_every = 60
    if total_frames > 0:
        print(f"[progress] 0/{total_frames} frames")
    else:
        print("[progress] 0/? frames")

    frame_id = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_out = frame.copy()
        frame_raw = frame.copy()

        try:
            # Cooldown: show warning screen and skip heavy processing.
            if lane_change_cooldown > 0:
                lane_change_cooldown -= 1
                _render_lane_change_screen(frame_raw, direction=lane_change_last_dir)
                _put_label(frame_raw, f"Frame {frame_id}", xy=(20, 50), scale=0.9, thickness=2)
                out.write(frame_raw)
                frame_id += 1
                if (frame_id % progress_every) == 0:
                    print(f"[progress] {frame_id}/{total_frames if total_frames > 0 else '?'} frames")
                continue

            # ROI (parameters resolved from curve_detection_pipeline defaults; local ROI helper kept for robustness).
            roi_bgr, roi_pts = apply_roi_mask(
                frame,
                top_y_ratio=top_y_ratio_f,
                left_bottom_ratio=left_bottom_ratio_f,
                right_bottom_ratio=right_bottom_ratio_f,
                top_left_x_ratio=top_left_x_ratio_f,
                top_right_x_ratio=top_right_x_ratio_f,
            )

            # Build asymmetric ALT ROIs (left/right) for color mask robustness on curves.
            roi_alt_L_bgr, roi_alt_R_bgr = _build_alt_rois(
                frame,
                roi_pts=roi_pts,
                top_y_ratio=top_y_ratio_f,
                left_bottom_ratio=left_bottom_ratio_f,
                right_bottom_ratio=right_bottom_ratio_f,
                top_left_x_ratio=top_left_x_ratio_f,
                top_right_x_ratio=top_right_x_ratio_f,
            )

            # COLOR masks on ALT ROIs.
            mask_hsv = cv2.bitwise_or(mask_hsv_curve_tuned(roi_alt_L_bgr), mask_hsv_curve_tuned(roi_alt_R_bgr))
            mask_hls = cv2.bitwise_or(
                mask_hls_clahe(roi_alt_L_bgr, clahe_clip_f, clahe_grid_i),
                mask_hls_clahe(roi_alt_R_bgr, clahe_clip_f, clahe_grid_i),
            )
            color_union = cv2.bitwise_or(mask_hls, mask_hsv)

            # Corridor for Hough and mask selection.
            hough_corridor = build_corridor_from_mask(color_union, dilate_px=merge_grad_dist_px_i)
            merged_hough_union = cv2.bitwise_and(color_union, color_union, mask=hough_corridor) if hough_corridor is not None else color_union

            best_name = "MERGED_HOUGH_UNION_COLOR"
            best_mask = merged_hough_union
            best_clean = clean_mask_light(best_mask)

            # Hough edges + lines.
            y_top_roi = roi_top_y_from_pts(roi_pts, h)
            y_top_draw = int(max(y_top_roi, int(h * overlay_top_y_ratio_f)))
            y_top_draw = int(np.clip(y_top_draw, y_top_roi, h - 2))

            hough_gate_mask = hough_corridor if hough_corridor is not None else best_clean
            y_top_keep_for_hough = int(max(0, int(0.85 * int(y_top_roi))))

            edges_for_hough_img = edges_for_hough(
                roi_bgr,
                mask_clean_01=hough_gate_mask,
                y_top_keep=y_top_keep_for_hough,
                clahe_clip=clahe_clip_f,
                clahe_grid=clahe_grid_i,
                canny1=30,
                canny2=100,
                gate_dilate_px=15,
            )

            lines = hough_lines_p(edges_for_hough_img)
            left_lines, right_lines = split_hough_lines_left_right(lines, w=w)

            # Lane-change detection from Hough center.
            y_fracs_lc = [0.72, 0.80, 0.88, 0.90]
            center_hough, width_hough, n_valid_hough = _hough_center_estimate_multi(
                left_lines, right_lines, w=w, h=h, y_fracs=y_fracs_lc
            )

            lane_change_dir = "Unknown"
            is_lane_change = False

            min_lines_per_side = 2
            conf_ok = (len(left_lines) >= min_lines_per_side) and (len(right_lines) >= min_lines_per_side)

            lc_dbg: Dict[str, float] = {
                "baseline": -1.0,
                "center": -1.0,
                "shift": 0.0,
                "thr": 0.0,
                "width": -1.0,
                "width_jump": 0.0,
                "width_ok": 0.0,
                "agree": 0.0,
                "conf_ok": 1.0 if conf_ok else 0.0,
                "n_valid": float(n_valid_hough),
            }

            if center_hough is not None and width_hough is not None and conf_ok:
                center_hough = float(center_hough)
                width_hough = float(width_hough)

                thr_high = max(30.0, 0.050 * float(w))
                thr_low = max(22.0, 0.035 * float(w))
                thr = thr_low if lane_change_streak > 0 else thr_high

                # Width sanity: keep ONLY a plausible absolute range check.
                # Do NOT gate lane-change on inter-frame width jumps, because during lane changes
                # the Hough width can vary substantially and would suppress detection.
                width_ok = (0.35 * float(w) <= width_hough <= 0.80 * float(w))
                width_jump = 0.0
                if prev_width is not None:
                    width_jump = abs(width_hough - float(prev_width))

                baseline_center: Optional[float] = None
                if len(baseline_centers) >= int(baseline_center_min_valid):
                    baseline_center = float(np.median(np.array(list(baseline_centers), dtype=np.float64)))

                shift = 0.0
                agree = 0
                is_lane_change_raw = False

                if baseline_center is not None:
                    shift = abs(center_hough - baseline_center)
                    for f in y_fracs_lc:
                        c1, _w1 = _hough_center_estimate(left_lines, right_lines, w=w, h=h, y_frac=float(f))
                        if c1 is None or (not np.isfinite(c1)):
                            continue
                        if abs(float(c1) - baseline_center) > thr:
                            agree += 1
                    # Disable width_ok gating for lane-change decision (keep it only for debugging/telemetry).
                    is_lane_change_raw = (shift > thr) and (agree >= 3)

                lane_change_streak = lane_change_streak + 1 if is_lane_change_raw else 0
                is_lane_change = lane_change_streak >= int(lane_change_required_streak)

                delta = (center_hough - baseline_center) if baseline_center is not None else 0.0
                if abs(delta) > 1e-6:
                    lane_change_dir = "Right" if delta > 0 else "Left"

                prev_width = float(width_hough)
                if lane_change_streak == 0:
                    baseline_centers.append(float(center_hough))

                lc_dbg.update(
                    {
                        "baseline": float(baseline_center) if baseline_center is not None else -1.0,
                        "center": float(center_hough),
                        "shift": float(shift),
                        "thr": float(thr),
                        "width": float(width_hough),
                        "width_jump": float(width_jump),
                        "width_ok": 1.0 if width_ok else 0.0,
                        "raw": 1.0 if is_lane_change_raw else 0.0,
                        "agree": float(agree),
                        "conf_ok": 1.0,
                        "n_valid": float(n_valid_hough),
                    }
                )
            else:
                lane_change_streak = 0

            if is_lane_change:
                lane_change_last_dir = lane_change_dir
                lane_change_cooldown = int(lane_change_cooldown_frames)
                lane_change_streak = 0

                # Reset smoothing so we don't drag old lane into new one.
                hist_left_x.clear()
                hist_right_x.clear()
                smooth_left_x = None
                smooth_right_x = None
                baseline_centers.clear()
                prev_width = None

                _render_lane_change_screen(frame_raw, direction=lane_change_last_dir)
                _put_label(frame_raw, f"Frame {frame_id}", xy=(20, 50), scale=0.9, thickness=2)
                out.write(frame_raw)
                frame_id += 1
                if (frame_id % progress_every) == 0:
                    print(f"[progress] {frame_id}/{total_frames if total_frames > 0 else '?'} frames")
                continue

            # Select pixels near Hough lines, then fit quadratic per side.
            xsL, ysL, xsR, ysR = select_lane_pixels_by_hough_proximity(
                best_clean, left_lines, right_lines, max_dist_px=hough_dist_px_f
            )

            haveL = len(left_lines) > 0
            haveR = len(right_lines) > 0

            if (xsL.size < 120 and haveL) or (xsR.size < 120 and haveR):
                xsL2, ysL2, xsR2, ysR2 = select_lane_pixels_by_hough_proximity(
                    best_clean, left_lines, right_lines, max_dist_px=hough_dist_px_f * 1.8
                )
                if xsL.size < 120 and haveL and xsL2.size > xsL.size:
                    xsL, ysL = xsL2, ysL2
                if xsR.size < 120 and haveR and xsR2.size > xsR.size:
                    xsR, ysR = xsR2, ysR2

            # Fallback only if a side has no lines at all.
            if (xsL.size < 120 and not haveL) or (xsR.size < 120 and not haveR):
                ys_all, xs_all = np.where(best_clean > 0)
                if xs_all.size >= 300:
                    mid = int(w // 2)
                    li = xs_all < mid
                    ri = ~li
                    if xsL.size < 120 and (not haveL) and np.count_nonzero(li) > 150:
                        xsL, ysL = xs_all[li].astype(np.int32), ys_all[li].astype(np.int32)
                    if xsR.size < 120 and (not haveR) and np.count_nonzero(ri) > 150:
                        xsR, ysR = xs_all[ri].astype(np.int32), ys_all[ri].astype(np.int32)

            left_fit = fit_poly_x_of_y(xsL, ysL, degree=2)
            right_fit = fit_poly_x_of_y(xsR, ysR, degree=2)

            # Smooth by sampling x(y), applying history-median + EMA, then refitting.
            y0 = int(max(0, min(int(y_top_draw), h - 2)))
            y1 = int(h - 1)
            K = 480
            ys_samp = np.linspace(y0, y1, K).astype(np.int32)

            def _xs_from_fit(fit: Optional[np.ndarray]) -> Optional[np.ndarray]:
                if fit is None:
                    return None
                try:
                    xs = np.polyval(fit, ys_samp.astype(np.float32)).astype(np.float64)
                    return np.clip(xs, 0.0, float(w - 1))
                except Exception:
                    return None

            cur_left_x = _xs_from_fit(left_fit)
            cur_right_x = _xs_from_fit(right_fit)

            if cur_left_x is not None:
                hist_left_x.append(cur_left_x)
            if cur_right_x is not None:
                hist_right_x.append(cur_right_x)

            med_left_x = _median_from_hist(hist_left_x)
            med_right_x = _median_from_hist(hist_right_x)

            smooth_left_x = _ema_update(smooth_left_x, med_left_x)
            smooth_right_x = _ema_update(smooth_right_x, med_right_x)

            left_fit_sm = _fit_from_xy(ys_samp, smooth_left_x, degree=2) if smooth_left_x is not None else None
            right_fit_sm = _fit_from_xy(ys_samp, smooth_right_x, degree=2) if smooth_right_x is not None else None

            if left_fit_sm is not None:
                left_fit = left_fit_sm
            if right_fit_sm is not None:
                right_fit = right_fit_sm

            if left_fit is None or right_fit is None:
                _put_label(frame_out, f"Curve: (no stable fit) | best={best_name}")
                out.write(frame_out)
                frame_id += 1
                if (frame_id % progress_every) == 0:
                    print(f"[progress] {frame_id}/{total_frames if total_frames > 0 else '?'} frames")
                continue

            # Optional geometry enforcement (TEST ONLY): enforce min/max lane width on the bottom band.
            # Default is OFF to preserve production behavior.
            if bool(geom_enforce_bottom_width):
                if not hasattr(cdp, "enforce_min_lane_width_bottom_band"):
                    raise RuntimeError(
                        "curve_detection_pipeline must expose enforce_min_lane_width_bottom_band(...)."
                    )

                def _bottom_band_mask(ys: np.ndarray) -> np.ndarray:
                    # Bottom band is the lowest `geom_bottom_band_ratio` portion of the sampled y-range.
                    y_min = float(np.min(ys))
                    y_max = float(np.max(ys))
                    y_band_top = y_max - float(geom_bottom_band_ratio) * (y_max - y_min)
                    return ys.astype(np.float64) >= float(y_band_top)

                # Evaluate current fits at the sampled ys.
                xL = np.polyval(left_fit, ys_samp.astype(np.float32)).astype(np.float64)
                xR = np.polyval(right_fit, ys_samp.astype(np.float32)).astype(np.float64)

                band_m = _bottom_band_mask(ys_samp)
                gap0 = (xR - xL)[band_m]
                if gap0.size > 0 and frame_id == 0:
                    print(
                        f"[geom] before: gap(min/med/max)={float(np.min(gap0)):.1f}/{float(np.median(gap0)):.1f}/{float(np.max(gap0)):.1f} px | "
                        f"minW={float(geom_min_width_ratio)*float(w):.1f}px "
                        f"maxW={(('None' if geom_max_width_ratio is None else f'{float(geom_max_width_ratio)*float(w):.1f}px'))} "
                        f"band={float(geom_bottom_band_ratio):.2f} margin={float(geom_margin_ratio):.3f}"
                        f" anchor_weight={float(geom_anchor_weight):.2f}"
                    )

                try:
                    xL_adj, xR_adj = cdp.enforce_min_lane_width_bottom_band(  # type: ignore[attr-defined]
                        xL,
                        xR,
                        ys_samp,
                        roi_pts=roi_pts,
                        w=int(w),
                        h=int(h),
                        y_top_draw=int(y_top_draw),
                        enabled=True,
                        bottom_band_ratio=float(geom_bottom_band_ratio),
                        margin_ratio=float(geom_margin_ratio),
                        min_width_ratio=float(geom_min_width_ratio),
                        max_width_ratio=(None if geom_max_width_ratio is None else float(geom_max_width_ratio)),
                        anchor_weight=float(geom_anchor_weight),
                        xsL_sel=xsL,
                        ysL_sel=ysL,
                        xsR_sel=xsR,
                        ysR_sel=ysR,
                    )
                except Exception as e:
                    if frame_id == 0:
                        print(f"[geom] ERROR: enforce_min_lane_width_bottom_band failed: {type(e).__name__}: {e}")
                    # Preserve production behavior: if enforcement fails, keep current fits.
                    xL_adj, xR_adj = xL, xR

                # === EMA smoothing for geometry-enforced bottom-band xL/xR to reduce flicker ===
                if geom_smooth_alpha > 0:
                    if smooth_geom_left is None:
                        smooth_geom_left = np.asarray(xL_adj, dtype=np.float64)
                        smooth_geom_right = np.asarray(xR_adj, dtype=np.float64)
                    else:
                        a = float(geom_smooth_alpha)
                        smooth_geom_left = (1.0 - a) * smooth_geom_left + a * np.asarray(xL_adj, dtype=np.float64)
                        smooth_geom_right = (1.0 - a) * smooth_geom_right + a * np.asarray(xR_adj, dtype=np.float64)

                    xL_adj = smooth_geom_left
                    xR_adj = smooth_geom_right

                gap1 = (np.asarray(xR_adj, dtype=np.float64) - np.asarray(xL_adj, dtype=np.float64))[band_m]
                if gap1.size > 0 and frame_id == 0:
                    print(
                        f"[geom] after : gap(min/med/max)={float(np.min(gap1)):.1f}/{float(np.median(gap1)):.1f}/{float(np.max(gap1)):.1f} px"
                    )

                # Refit from adjusted samples. Only accept if finite.
                left_fit_g = np.polyfit(ys_samp.astype(np.float32), np.asarray(xL_adj, dtype=np.float32), 2)
                right_fit_g = np.polyfit(ys_samp.astype(np.float32), np.asarray(xR_adj, dtype=np.float32), 2)
                if np.all(np.isfinite(left_fit_g)) and np.all(np.isfinite(right_fit_g)):
                    left_fit = left_fit_g
                    right_fit = right_fit_g

            poly_pts, left_curve, right_curve = build_lane_polygon_from_fits(
                left_fit,
                right_fit,
                roi_pts=roi_pts,
                h=int(h),
                w=int(w),
                y_top_draw=int(y_top_draw),
            )
            if poly_pts is None:
                _put_label(frame_out, f"Curve: (bad polygon) | best={best_name}")
                out.write(frame_out)
                frame_id += 1
                if (frame_id % progress_every) == 0:
                    print(f"[progress] {frame_id}/{total_frames if total_frames > 0 else '?'} frames")
                continue

            overlay = frame_out.copy()
            cv2.fillPoly(overlay, [poly_pts], (0, 255, 0))
            cv2.polylines(overlay, [left_curve], False, (255, 0, 255), 6, cv2.LINE_AA)
            cv2.polylines(overlay, [right_curve], False, (0, 255, 255), 6, cv2.LINE_AA)
            cv2.addWeighted(overlay, lane_fill_alpha_f, frame_out, 1.0 - lane_fill_alpha_f, 0.0, frame_out)

            # Curvature estimate (meters) from quadratic term.
            y_eval = float(h - 1)

            def _radius_meters(fit: np.ndarray) -> float:
                a, b, c = float(fit[0]), float(fit[1]), float(fit[2])
                a_m = (xm_per_pix / (ym_per_pix ** 2)) * a
                b_m = (xm_per_pix / ym_per_pix) * b
                coeffs_m = np.array([a_m, b_m, xm_per_pix * c], dtype=np.float64)
                y_eval_m = y_eval * ym_per_pix

                a2 = float(coeffs_m[0])
                b2 = float(coeffs_m[1])
                dx_dy = 2.0 * a2 * y_eval_m + b2
                d2x = 2.0 * a2
                if abs(d2x) < 1e-10:
                    return float("inf")
                return ((1.0 + dx_dy * dx_dy) ** 1.5) / abs(d2x)

            R_left = _radius_meters(left_fit)
            R_right = _radius_meters(right_fit)

            direction = "Straight"
            a_avg = 0.5 * (float(left_fit[0]) + float(right_fit[0]))
            if abs(a_avg) > 2.5e-7:
                direction = "Left" if a_avg < 0 else "Right"

            def _fmt_R(R: float) -> str:
                if (not np.isfinite(R)) or R > 1e6:
                    return "inf"
                return f"{R:,.0f}m"

            _put_label(
                frame_out,
                (
                    f"LC dbg: base={lc_dbg.get('baseline', -1.0):.1f} center={lc_dbg.get('center', -1.0):.1f} "
                    f"shift={lc_dbg.get('shift', 0.0):.1f} thr={lc_dbg.get('thr', 0.0):.1f} "
                    f"agree={lc_dbg.get('agree', 0.0):.0f} conf={lc_dbg.get('conf_ok', 0.0):.0f} n={lc_dbg.get('n_valid', 0.0):.0f} "
                    f"width={lc_dbg.get('width', -1.0):.1f} widthJump={lc_dbg.get('width_jump', 0.0):.1f} "
                    f"widthOk={lc_dbg.get('width_ok', 0.0):.0f} raw={lc_dbg.get('raw', 0.0):.0f} "
                    f"streak={lane_change_streak}/{lane_change_required_streak}"
                ),
                xy=(20, 160),
                scale=0.75,
                thickness=2,
            )

            _put_label(frame_out, f"Frame {frame_id} | Curve: {direction} | best={best_name}")
            _put_label(
                frame_out,
                f"R_left={_fmt_R(R_left)}  R_right={_fmt_R(R_right)} | deg=2 | lc_y=0.90h",
                xy=(20, 92),
                scale=0.8,
                thickness=2,
            )

        except Exception as e:
            # Never crash the whole run - annotate the frame and continue.
            err = f"ERROR: {type(e).__name__}: {e}"
            _put_label(frame_raw, err, xy=(20, 200), scale=0.7, thickness=2)
            out.write(frame_raw)
            frame_id += 1
            if (frame_id % progress_every) == 0:
                print(f"[progress] {frame_id}/{total_frames if total_frames > 0 else '?'} frames")
            continue

        out.write(frame_out)
        frame_id += 1
        if (frame_id % progress_every) == 0:
            print(f"[progress] {frame_id}/{total_frames if total_frames > 0 else '?'} frames")

    cap.release()
    out.release()
    if total_frames > 0:
        print(f"[progress] done: {frame_id}/{total_frames} frames")
    else:
        print(f"[progress] done: {frame_id} frames")


def main() -> None:
    """
    Main entry point for curve detection video processing.
    
    Parses command-line arguments and runs the curve detection pipeline.
    """
    p = argparse.ArgumentParser()

    # Default input under <project_root>/data/processed/
    default_video = os.path.join(_PROJECT_ROOT, "data", "processed", "Curve_Detection_Input.mp4")

    p.add_argument("--video", default=default_video, help="Input video path")
    p.add_argument(
        "--output",
        default="",
        help="Output video path. If empty, uses <lane-detection>/output/<input_basename>_output.mp4",
    )

    # Geometry enforcement (TEST ONLY) - OFF by default
    p.add_argument("--geom_enforce_bottom_width", action="store_true")
    p.add_argument("--geom_min_width_ratio", type=float, default=0.22)
    p.add_argument(
        "--geom_max_width_ratio",
        type=float,
        default=None,
        help="Optional maximum lane width on the enforced bottom band, as a ratio of image width (e.g., 0.40).",
    )
    p.add_argument("--geom_bottom_band_ratio", type=float, default=0.30)
    p.add_argument("--geom_margin_ratio", type=float, default=0.02)
    p.add_argument(
        "--geom_anchor_weight",
        type=float,
        default=0.0,
        help=(
            "Bias width enforcement to keep one side fixed when expanding/shrinking on the bottom band. "
            "+ values keep RIGHT fixed; - values keep LEFT fixed; 0 = symmetric."
        ),
    )
    p.add_argument(
        "--geom_smooth_alpha",
        type=float,
        default=0.18,
        help="EMA smoothing factor applied to geometry-enforced bottom-band xL/xR to reduce flicker (0 disables).",
    )

    args = p.parse_args()

    def _resolve_video_path(user_path: str) -> str:
        # 1) If user gave an absolute existing path - use it.
        p_abs = os.path.abspath(user_path)
        if os.path.isabs(user_path) and os.path.exists(p_abs):
            return p_abs

        # 2) If user gave a relative path, prefer resolving from repo root.
        #    Example: data/processed/<clip>.mp4
        cand_repo = os.path.abspath(os.path.join(_PROJECT_ROOT, user_path))
        if os.path.exists(cand_repo):
            return cand_repo

        # 3) Common mistake: pointing under lane-detection/data/processed/... but videos live under repo_root/data/processed/...
        #    If the basename exists under repo_root/data/processed, use that.
        base = os.path.basename(p_abs)
        cand_processed = os.path.abspath(os.path.join(_PROJECT_ROOT, "data", "processed", base))
        if os.path.exists(cand_processed):
            return cand_processed

        # Fall through - keep the original absolute for a clear error message.
        return p_abs

    video_path = _resolve_video_path(args.video)
    if not args.output:
        in_base = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(_LANE_DETECTION_ROOT, "output", f"{in_base}_output.mp4")
    else:
        output_path = os.path.abspath(args.output)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("🎬 Curve detection runner")
    print(f"   input : {video_path}")
    if not os.path.exists(video_path):
        print(f"   NOTE: file not found at: {video_path}")
        print(f"   TIP : put videos under: {os.path.join(_PROJECT_ROOT, 'data', 'processed')} or pass a correct --video path")
    print(f"   output: {output_path}")
    print(
        "   geom_enforce_bottom_width:",
        bool(args.geom_enforce_bottom_width),
        "min_width_ratio=", float(args.geom_min_width_ratio),
        "max_width_ratio=", args.geom_max_width_ratio,
        "bottom_band_ratio=", float(args.geom_bottom_band_ratio),
        "margin_ratio=", float(args.geom_margin_ratio),
        "anchor_weight=", float(args.geom_anchor_weight),
    )

    process_video_with_curve_prediction(
        video_path=video_path,
        output_path=output_path,
        geom_enforce_bottom_width=bool(args.geom_enforce_bottom_width),
        geom_min_width_ratio=float(args.geom_min_width_ratio),
        geom_max_width_ratio=args.geom_max_width_ratio,
        geom_bottom_band_ratio=float(args.geom_bottom_band_ratio),
        geom_margin_ratio=float(args.geom_margin_ratio),
        geom_anchor_weight=float(args.geom_anchor_weight),
        geom_smooth_alpha=float(args.geom_smooth_alpha),
    )


if __name__ == "__main__":
    main()