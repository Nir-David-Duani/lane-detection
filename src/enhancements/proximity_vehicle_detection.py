"""
Proximity-Based Vehicle Detection for Collision Avoidance
=========================================================

Part 2 – Enhancement: Proximity-Based Vehicle Detection
-------------------------------------------------------

Objective
~~~~~~~~~
Integrate a feature that detects **nearby vehicles** in the scene to aid
in basic collision avoidance. The focus is on identifying vehicles that
are in close proximity to the ego-vehicle (the camera car).

Assignment Requirements (summary)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Use *classical* computer vision techniques for vehicle detection
  (no deep learning requirement), for example:
  - template matching,
  - feature-based detection,
  - morphological operations and shape/size heuristics.
- For each detected nearby vehicle, estimate its **proximity in meters**
  (using projective-geometry heuristics, lane width assumptions, or a
  simple camera calibration).
- Visually indicate all vehicles that are close to the driver’s car and
  may pose a collision risk in the output video by:
  - drawing **bounding boxes** around them, and
  - overlaying a text label with the estimated distance (e.g. ``"12 m"``).
  This should work for **all close cars in the frame**, not just one.

Relation to the Existing Lane Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This enhancement is designed to be layered **on top of** the existing
lane detection pipeline:

- It must reuse functions from :mod:`pipeline` to obtain:
  - a Region of Interest over the road
    (:func:`pipeline.apply_roi_mask`),
  - a lane-focused color / edge representation
    (:func:`pipeline.apply_color_threshold`,
     :func:`pipeline.apply_canny`,
     :func:`pipeline.detect_lines_hough` as needed).
- The lane information can be used to:
  - Restrict the search area for vehicles (e.g., near/inside the lane),
  - Approximate perspective / scale to reason about proximity
    (vehicles lower in the frame are closer, etc.).

Module Responsibilities
~~~~~~~~~~~~~~~~~~~~~~~
- Provide a processing function that:
  - reads video frames,
  - calls the pre-processing steps from :mod:`pipeline`,
  - applies additional CV logic to **detect vehicles**,
  - estimates which detections are "nearby" / potentially dangerous,
  - and draws bounding boxes or overlays to visualize them.
- Ensure that all Part 1 requirements (lane change detection, etc.)
  remain satisfied when this enhancement is active (by integrating with
  the logic in :mod:`line_detection` rather than re-implementing it).

Implementation Sketch (to be completed)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Define a public API function, for example:
  
  - ``process_video_with_vehicle_proximity(video_path, output_path, ...)``
  
  which will:
  
  1. Read frames from the input video.
  2. Use :mod:`pipeline` to obtain a road-focused ROI and lane context.
  3. Run a classical-CV-based vehicle detection algorithm within the
     relevant regions.
  4. Estimate proximity (e.g., via bounding box size/position, or a
     simple heuristic based on image coordinates).
  5. Draw bounding boxes / warnings on vehicles in dangerous proximity.
  6. Write the annotated output video.

At this stage this file intentionally contains **documentation only**.
The actual implementation should be added in a later step, making sure
to reuse the existing functions from :mod:`pipeline` instead of
duplicating pre-processing logic.
"""


