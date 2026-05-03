"""
MediaPipe-based pose analyzer.

This module exists as a parallel implementation alongside the existing YOLO
pose detection. The YOLO path (in main.py) gives 17 keypoints with heel and
toe always None — adequate for ankle-based metrics but insufficient for
detecting genuine ground-contact events (which need toe Y trajectory).

MediaPipe Pose Landmarker (full model) gives all 33 keypoints with heel and
toe populated. Visibility scores are reliably > 0.9 for foot keypoints in
side-on bowling video.

The output shape MATCHES the YOLO path output so the frontend can consume
either without code changes — just routes to /analyze-mp instead of /analyze.

This file is loaded lazily (at endpoint call time) so that if MediaPipe
fails to import for any reason, the existing /analyze endpoint still works.
"""
from __future__ import annotations
import os
import sys
import time
import urllib.request
from typing import Any

import cv2
import numpy as np

# MediaPipe model: full Pose Landmarker. ~12 MB.
# Cached locally on first use to avoid re-downloading on every container start.
MP_MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
    'pose_landmarker_full/float16/latest/pose_landmarker_full.task'
)
MP_MODEL_PATH = '/tmp/pose_landmarker_full.task'

# Lazily-imported MediaPipe references; populated on first use.
_mp = None
_vision = None
_BaseOptions = None


def _ensure_mediapipe_loaded():
    """Import MediaPipe and download the model on first use.

    Raised exceptions propagate to caller so the endpoint can return a clean
    500 with a useful message rather than crashing the worker.
    """
    global _mp, _vision, _BaseOptions
    if _mp is not None:
        return

    import mediapipe as mp_mod
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision

    _mp = mp_mod
    _vision = mp_vision
    _BaseOptions = mp_tasks.BaseOptions

    if not os.path.exists(MP_MODEL_PATH):
        print(f'[mp] Downloading model to {MP_MODEL_PATH} (~12 MB)...', flush=True)
        urllib.request.urlretrieve(MP_MODEL_URL, MP_MODEL_PATH)
        print(f'[mp] Model downloaded.', flush=True)


def _detect_slomo_basic(video_path: str, override_factor: float | None = None) -> float:
    """Lightweight slo-mo detection. Mirrors what the YOLO path does but
    we don't need its complexity here — Railway calls us with fps_hint
    that already encodes the user's video-type selection."""
    if override_factor:
        return float(override_factor)
    return 1.0


def analyze_video_mp(
    video_path: str,
    fps_hint: float,
    height_cm: float | None = None,
    bowling_arm: str = 'right',
    progress_cb=None,
) -> dict[str, Any]:
    """Run MediaPipe pose detection on a video.

    Returns a dict with the same shape as the YOLO path:
        {
            'video_info': {...},
            'frames': [
                {'frame': int, 'time': float, 'keypoints': [...], 'confidence': float},
                ...
            ],
        }

    Each frame's `keypoints` is a list of 33 dicts with x, y, z, visibility —
    or None if no pose was detected in that frame.

    progress_cb is an optional callable(frames_done: int, total: int) for
    incremental progress reporting.
    """
    _ensure_mediapipe_loaded()

    # Slo-mo handling matches the YOLO path's logic at a high level.
    override_factor = None
    if fps_hint and fps_hint > 30:
        override_factor = fps_hint / 30
    slomo_factor = _detect_slomo_basic(video_path, override_factor)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f'Could not open video: {video_path}')

    video_fps = cap.get(cv2.CAP_PROP_FPS) or fps_hint or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Frame skipping: same target_real_fps as the YOLO path so the analyzed
    # frame indices line up between modes (makes side-by-side comparison clean)
    target_real_fps = 120
    if slomo_factor > 1:
        skip = max(1, round(video_fps * slomo_factor / target_real_fps))
    elif video_fps <= 30:
        skip = 2
    elif video_fps <= 60:
        skip = 3
    elif video_fps <= 120:
        skip = 6
    else:
        skip = 12

    analysis_fps = video_fps / skip
    total_analysis = total_frames // skip

    print(f'[mp] Video: {width}×{height} @ {video_fps}fps, slomo={slomo_factor}x, '
          f'skip={skip}, total_analysis={total_analysis}', flush=True)

    # Build the detector. VIDEO mode benefits from temporal tracking, which
    # is critical for stable left/right limb labelling in profile shots.
    base_options = _BaseOptions(model_asset_path=MP_MODEL_PATH)
    options = _vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = _vision.PoseLandmarker.create_from_options(options)

    frames_out = []
    valid_count = 0
    t0 = time.time()
    target_frame = 0
    frames_done = 0

    try:
        while target_frame < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ok, frame = cap.read()
            if not ok:
                break

            time_s = target_frame / video_fps / slomo_factor
            timestamp_ms = int(target_frame * 1000.0 / video_fps)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb)
            res = detector.detect_for_video(mp_image, timestamp_ms)

            if res.pose_landmarks and len(res.pose_landmarks) > 0:
                lms = res.pose_landmarks[0]
                # MediaPipe normalizes to 0-1; multiply by image dimensions
                kp = [
                    {
                        'x': round(lm.x * width, 1),
                        'y': round(lm.y * height, 1),
                        'z': round(lm.z, 4),
                        'visibility': round(lm.visibility, 3),
                    }
                    for lm in lms
                ]
                avg_v = sum(p['visibility'] for p in kp) / len(kp)
                frames_out.append({
                    'frame': target_frame,
                    'time': round(time_s, 4),
                    'keypoints': kp,
                    'cropped_keypoints': None,  # MediaPipe doesn't need crop-zoom; full-frame is high-quality
                    'confidence': round(avg_v, 3),
                })
                valid_count += 1
            else:
                frames_out.append({
                    'frame': target_frame,
                    'time': round(time_s, 4),
                    'keypoints': None,
                    'cropped_keypoints': None,
                    'confidence': 0.0,
                })

            frames_done += 1
            if progress_cb:
                progress_cb(frames_done, total_analysis)
            target_frame += skip
    finally:
        detector.close()
        cap.release()

    elapsed = time.time() - t0
    print(f'[mp] Analysed {frames_done} frames ({valid_count} with valid pose) '
          f'in {elapsed:.1f}s = {frames_done/max(elapsed,0.01):.1f} fps', flush=True)

    return {
        'video_info': {
            'width': width,
            'height': height,
            'fps': round(video_fps, 2),
            'total_frames': total_frames,
            'duration': round(total_frames / video_fps, 3) if video_fps > 0 else 0,
            'analysis_fps': round(analysis_fps, 1),
            'effective_fps': round(analysis_fps * slomo_factor, 1),
            'slomo_factor': slomo_factor,
            'real_duration': round(total_frames / video_fps / slomo_factor, 3) if video_fps > 0 else 0,
            'analyzer': 'mediapipe-pose-full',
        },
        'frames': frames_out,
        'ball_speed': None,  # MediaPipe path doesn't do ball-speed detection (yet)
    }
