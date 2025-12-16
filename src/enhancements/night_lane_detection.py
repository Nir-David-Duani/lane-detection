"""
Night-Time Lane Detection Enhancement
=====================================

Part 2 – Enhancement: Night-Time Lane Detection
-----------------------------------------------

Objective
~~~~~~~~~
Adapt the existing lane detection algorithm to work effectively in
**low-light / night-time conditions**, where lane markings are less visible
and the overall appearance of the scene is darker and noisier.

Key constraints from the assignment:

- This enhancement **replaces the requirement for a daytime video** –
  the evaluation video may be night-time only.
- However, the **resulting code must be unified**:
  it should work on both **day and night** videos.
  (No separate duplicated implementations for day and night.)

Module Responsibilities
~~~~~~~~~~~~~~~~~~~~~~~
This module is responsible for adding *illumination-robust* pre-processing
on top of the existing lane pipeline, while reusing **as much existing code
as possible**. In particular, it must:

- Reuse the core stages implemented in :mod:`pipeline` **without
  re-implementing them**:
  - :func:`pipeline.apply_roi_mask`
  - :func:`pipeline.apply_color_threshold`
  - :func:`pipeline.apply_canny`
  - :func:`pipeline.detect_lines_hough`
- Reuse the higher-level lane logic from :mod:`line_detection` as much
  as possible (e.g. :class:`line_detection.LaneChangeDetector`,
  :func:`line_detection.filter_lines_by_slope`,
  :func:`line_detection.fit_lane_line`,
  :func:`line_detection.extrapolate_line`,
  :func:`line_detection.draw_lines`), instead of writing new variants.
- Introduce additional image enhancement steps *before* or *around* the
  existing pipeline to improve lane visibility at night, e.g.:
  - brightness/contrast adjustment,
  - gamma correction,
  - histogram equalization (global or CLAHE),
  - denoising / smoothing adapted for low light.
- Keep the **same logical pipeline structure** as in :mod:`line_detection`
  (lane change detection, smoothing, visualization), but applied to
  enhanced frames so that the same code path works for both day and night.

Expected Behaviour
~~~~~~~~~~~~~~~~~~
- On **daytime videos** the enhancement should not significantly degrade
  performance (ideally should be a no-op or mildly improve robustness).
- On **night-time videos** the enhancement should:
  - make lane markings more visible,
  - enable the standard lane detection pipeline (from :mod:`pipeline`)
    to still detect lanes with accuracy comparable to the daytime case.

Implementation Sketch (to be completed)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Define a public API function, e.g.:
  
  - ``process_video_night_robust(video_path, output_path, ...)``
  
  which will:
  
  1. Read frames from the input video.
  2. Apply night-time specific enhancement to the frame.
  3. Run the common pre-processing stages by calling functions from
     :mod:`pipeline`.
  4. Use or adapt the lane post-processing logic from :mod:`line_detection`
     (lane change detection, smoothing, drawing).
  5. Write an annotated output video.

At this stage this file intentionally contains **documentation only**.
The actual implementation should be added later, strictly reusing the
functions from :mod:`pipeline` for the core pre-processing steps.
"""


