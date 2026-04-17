"""Cricket Bowling Pose Analysis API — Async Job Queue version.

Changes from previous version:
- POST /analyze now returns immediately with a job_id
- Processing happens in a background thread
- GET /status/{job_id} returns progress (frontend polls this)
- GET /results/{job_id} returns full frame data when done
- ONNX input reduced to 416px for faster inference
- Frame skipping for 30fps video (every 2nd frame)
"""
import os
import math
import uuid
import time
import tempfile
import threading
import numpy as np
import cv2
import onnxruntime as ort
import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="Cricket Analyze Pro - Pose API")

# ─── Payment routes ──────────────────────────────────────────────
from payments import router as payments_router, init_stripe
app.include_router(payments_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://cricket-analyze-pro.vercel.app",
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_origin_regex=r"https://cricket-analyze-pro.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_URL = "https://huggingface.co/Xenova/yolov8-pose-onnx/resolve/main/yolov8n-pose.onnx"
MODEL_PATH = "/app/yolov8n-pose.onnx"
BALL_MODEL_PATH = "/app/cricket-ball-yolov11.onnx"  # Custom trained cricket ball detection model
INPUT_SIZE = 640
session = None
ball_session = None  # Will be loaded when custom model is deployed

# Roboflow hosted model for cricket ball detection
ROBOFLOW_API_KEY = "Ve9LS6zfaXWJp8IQCDnA"
ROBOFLOW_MODEL_ID = "cricket-dataset-z2wkt-ko5pz/1"
# Try serverless first (for YOLOv11 models), fall back to detect API
ROBOFLOW_URLS = [
    f"https://serverless.roboflow.com/{ROBOFLOW_MODEL_ID}",
    f"https://detect.roboflow.com/{ROBOFLOW_MODEL_ID}",
]
ROBOFLOW_URL = ROBOFLOW_URLS[0]  # Will be updated on first successful call


def detect_ball_roboflow_single(tile, conf_thresh=0.15):
    """Send a single image tile to Roboflow and return raw predictions."""
    global ROBOFLOW_URL
    _, img_encoded = cv2.imencode('.jpg', tile, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_bytes = img_encoded.tobytes()

    urls_to_try = [ROBOFLOW_URL] + [u for u in ROBOFLOW_URLS if u != ROBOFLOW_URL]
    for url in urls_to_try:
        try:
            resp = requests.post(
                url,
                params={
                    "api_key": ROBOFLOW_API_KEY,
                    "confidence": int(conf_thresh * 100),
                    "overlap": 30,
                },
                files={"file": ("frame.jpg", img_bytes, "image/jpeg")},
                timeout=15,
            )
            if resp.status_code == 200:
                if url != ROBOFLOW_URL:
                    print(f"Roboflow: switching to working URL: {url}")
                    ROBOFLOW_URL = url
                return resp.json().get("predictions", [])
            else:
                print(f"Roboflow {url}: HTTP {resp.status_code}")
        except requests.exceptions.Timeout:
            print(f"Roboflow {url}: timeout")
        except Exception as e:
            print(f"Roboflow {url}: {e}")
    return []


def detect_ball_roboflow(frame, conf_thresh=0.15):
    """Detect cricket ball using TILED approach for tall portrait frames.
    
    YOLO resizes input to 640x640. A 1080x1920 portrait frame gets compressed 3x vertically,
    making a 15px ball into 5px — undetectable. Tiling keeps the ball at detectable size.
    """
    try:
        full_h, full_w = frame.shape[:2]
        
        # Decide if tiling is needed: if height > 1.5x width, tile vertically
        if full_h > full_w * 1.5:
            # Create overlapping horizontal strips
            # Each tile should be roughly square for best YOLO performance
            tile_h = full_w  # make tiles square-ish
            overlap = tile_h // 3  # 33% overlap to catch ball on tile boundaries
            
            tiles = []
            y = 0
            while y < full_h:
                y_end = min(y + tile_h, full_h)
                # Don't create tiny leftover tiles
                if full_h - y_end < tile_h // 3 and y > 0:
                    y_end = full_h
                tiles.append((y, y_end))
                if y_end >= full_h:
                    break
                y += tile_h - overlap
        else:
            # Frame is already roughly square, send as-is
            tiles = [(0, full_h)]
        
        all_detections = []
        for tile_y1, tile_y2 in tiles:
            tile = frame[tile_y1:tile_y2, :]
            preds = detect_ball_roboflow_single(tile, conf_thresh)
            
            tile_h_actual = tile_y2 - tile_y1
            for pred in preds:
                pred_class = pred.get("class", "").lower()
                if pred_class and pred_class != "ball":
                    continue
                # Map tile coordinates back to full frame
                cx = pred.get("x", 0)
                cy = pred.get("y", 0) + tile_y1  # offset by tile position
                conf = pred.get("confidence", 0)
                all_detections.append({
                    "nx": round(cx / full_w, 4),
                    "ny": round(cy / full_h, 4),
                    "confidence": round(conf, 3),
                    "x": round(cx, 1),
                    "y": round(cy, 1),
                    "w": round(pred.get("width", 0), 1),
                    "h": round(pred.get("height", 0), 1),
                })
        
        # Deduplicate detections from overlapping tiles (same ball detected in 2 tiles)
        if len(all_detections) > 1:
            unique = [all_detections[0]]
            for det in all_detections[1:]:
                is_dup = False
                for u in unique:
                    if abs(det["nx"] - u["nx"]) < 0.03 and abs(det["ny"] - u["ny"]) < 0.03:
                        # Keep higher confidence
                        if det["confidence"] > u["confidence"]:
                            unique.remove(u)
                            unique.append(det)
                        is_dup = True
                        break
                if not is_dup:
                    unique.append(det)
            all_detections = unique
        
        all_detections.sort(key=lambda d: d["confidence"], reverse=True)
        return all_detections[:5]
    except Exception as e:
        print(f"Roboflow detection error: {e}")
        return []

# In-memory job store (fine for single-instance Railway deployment)
jobs = {}

# Auto-cleanup: remove jobs older than 30 minutes
def cleanup_jobs():
    cutoff = time.time() - 1800
    expired = [jid for jid, j in jobs.items() if j.get("created_at", 0) < cutoff]
    for jid in expired:
        # Clean up temp video file if it still exists
        tmp = jobs[jid].get("tmp_path")
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except Exception:
                pass
        del jobs[jid]


def download_model():
    # YOLOv8-nano pose is ~13.5MB. If the cached file is much larger (e.g. 46MB small model),
    # it's the wrong model and must be re-downloaded.
    EXPECTED_SIZE_MB = 13.5  # nano model
    MAX_VALID_SIZE_MB = 20   # anything over this is NOT nano
    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / 1e6
        if size_mb > 1 and size_mb < MAX_VALID_SIZE_MB:
            print(f"Pose model already exists: {size_mb:.1f} MB ({MODEL_PATH}) ✓ nano")
            return
        else:
            print(f"⚠️ Model file wrong size: {size_mb:.1f}MB (expected ~{EXPECTED_SIZE_MB}MB nano, max {MAX_VALID_SIZE_MB}MB) — re-downloading")
            os.remove(MODEL_PATH)
    print(f"Downloading pose model from {MODEL_URL}...")
    r = requests.get(MODEL_URL, allow_redirects=True, stream=True)
    r.raise_for_status()
    with open(MODEL_PATH, 'wb') as f:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
    size = os.path.getsize(MODEL_PATH)
    print(f"Pose model downloaded: {size/1e6:.1f} MB")
    if size < 1000000:
        raise RuntimeError(f"Model file too small ({size} bytes) — download may have failed")
    if size / 1e6 > MAX_VALID_SIZE_MB:
        raise RuntimeError(f"Model file too large ({size/1e6:.1f}MB) — expected nano (~{EXPECTED_SIZE_MB}MB). Check MODEL_URL.")


def preprocess(frame, input_size=INPUT_SIZE):
    h, w = frame.shape[:2]
    scale = min(input_size / w, input_size / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh))
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    padded[:nh, :nw] = resized
    blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(blob, 0), scale


def detect_pose_tiled(frame, session_ref, input_size=INPUT_SIZE, conf_thresh=0.25):
    """UNUSED — kept for reference. See detect_pose_cropped()."""
    pass


def detect_pose_cropped(frame, session_ref, input_size=INPUT_SIZE, conf_thresh=0.25):
    """Two-pass pose detection: detect bowler, crop, re-detect at high resolution.
    
    Problem: 1920×1080 → 640×360 makes bowler only 70px tall. Joints at 2-3px.
    
    Fix: 
      Pass 1: Full-frame detection → find bowler's bounding box
      Pass 2: Crop to bowler + padding → re-run at 640×640 on just the bowler
      Result: bowler fills most of the 640px model → joints at 15-20px
    
    This gives ~4x better joint accuracy for angles while preserving
    correct full-frame coordinates for stride/distance measurements.
    """
    h, w = frame.shape[:2]
    input_name = session_ref.get_inputs()[0].name
    
    # Pass 1: Full-frame detection to find bowler bounding box
    blob, scale = preprocess(frame, input_size)
    outputs = session_ref.run(None, {input_name: blob})
    landmarks_full, conf_full = postprocess(outputs, scale, conf_thresh)
    
    if not landmarks_full:
        return None, 0
    
    # Extract bounding box from detected keypoints
    xs = [lm["x"] for lm in landmarks_full if lm is not None]
    ys = [lm["y"] for lm in landmarks_full if lm is not None]
    if len(xs) < 4:
        return landmarks_full, conf_full  # not enough points, use pass 1
    
    # Bounding box with generous padding (50% each side)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    box_w = max_x - min_x
    box_h = max_y - min_y
    pad_x = box_w * 0.5
    pad_y = box_h * 0.4
    
    crop_x1 = max(0, int(min_x - pad_x))
    crop_y1 = max(0, int(min_y - pad_y))
    crop_x2 = min(w, int(max_x + pad_x))
    crop_y2 = min(h, int(max_y + pad_y))
    
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1
    
    # Only do pass 2 if cropping meaningfully increases resolution
    full_scale = min(input_size / w, input_size / h)
    crop_scale = min(input_size / crop_w, input_size / crop_h)
    if crop_scale < full_scale * 1.5:
        return landmarks_full, conf_full  # crop isn't much better
    
    # Pass 2: Crop and re-detect at higher resolution
    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    blob2, scale2 = preprocess(crop, input_size)
    outputs2 = session_ref.run(None, {input_name: blob2})
    landmarks_crop, conf_crop = postprocess(outputs2, scale2, conf_thresh)
    
    if not landmarks_crop or conf_crop < conf_full * 0.7:
        return landmarks_full, conf_full  # crop detection worse, use pass 1
    
    # Map crop coordinates back to full frame
    mapped = [None] * 33
    for idx, lm in enumerate(landmarks_crop):
        if lm is not None:
            mapped[idx] = {
                "x": round(lm["x"] + crop_x1, 1),
                "y": round(lm["y"] + crop_y1, 1),
                "z": lm["z"],
                "visibility": lm["visibility"],
            }
    
    return mapped, conf_crop


def postprocess(output, scale, conf_thresh=0.25):
    predictions = output[0]
    if len(predictions.shape) == 3:
        predictions = predictions[0].T
    if len(predictions) == 0:
        return None, 0
    scores = predictions[:, 4]
    mask = scores > conf_thresh
    predictions = predictions[mask]
    if len(predictions) == 0:
        return None, 0
    best_idx = predictions[:, 4].argmax()
    det = predictions[best_idx]
    kps_raw = det[5:].reshape(17, 3)
    kps_raw[:, 0] /= scale
    kps_raw[:, 1] /= scale
    COCO_TO_MP = {
        0: 0, 5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16,
        11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28,
    }
    mp_landmarks = [None] * 33
    total_conf = 0
    count = 0
    for coco_idx, mp_idx in COCO_TO_MP.items():
        x, y, conf = kps_raw[coco_idx]
        mp_landmarks[mp_idx] = {
            "x": round(float(x), 1),
            "y": round(float(y), 1),
            "z": 0,
            "visibility": round(float(conf), 3),
        }
        total_conf += conf
        count += 1
    return mp_landmarks, round(total_conf / max(count, 1), 3)


def detect_slomo(path, override_factor=None):
    """Detect iPhone slo-mo videos and return the speedup factor needed.
    
    CRITICAL: time_base=1/2400 is timestamp precision, NOT native fps.
    iPhone shoots slo-mo at BOTH 120fps and 240fps, both use time_base=1/2400.
    We can't reliably distinguish them from metadata alone.
    
    Strategy: try both factors, pick the one giving a reasonable real duration
    for a bowling delivery (2-7 seconds). Allow user override.
    """
    if override_factor and override_factor > 1:
        print(f"Slo-mo: using user override factor={override_factor}")
        return override_factor
        
    try:
        import subprocess, json
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", path],
            capture_output=True, text=True, timeout=10
        )
        data = json.loads(result.stdout)

        # Check for Apple slo-mo indicator
        tags = data.get("format", {}).get("tags", {})
        full_rate = tags.get("com.apple.quicktime.full-frame-rate-playback-intent", "1")

        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                time_base = stream.get("time_base", "")
                r_frame_rate = stream.get("r_frame_rate", "0/1")
                duration = float(stream.get("duration", 0))
                tb_denom = int(time_base.split("/")[1]) if "/" in time_base else 0
                num, den = r_frame_rate.split("/") if "/" in r_frame_rate else ("0", "1")
                display_fps = int(num) / max(int(den), 1)

                if tb_denom >= 240 and display_fps < 120 and full_rate == "0":
                    # It's slo-mo, but is it 120fps (4x) or 240fps (8x)?
                    # Try both and pick the one giving reasonable bowling duration
                    candidates = []
                    for native in [120, 240]:
                        factor = native / display_fps
                        real_duration = duration / factor
                        # A bowling delivery video should be 1.5-8 seconds real time
                        # Score: closer to 3-5 seconds = more likely correct
                        if real_duration > 0:
                            dist_from_ideal = abs(real_duration - 4.0)
                            candidates.append((factor, native, real_duration, dist_from_ideal))
                            print(f"Slo-mo candidate: {native}fps → {factor}x → {real_duration:.1f}s real duration")
                    
                    if candidates:
                        # Pick the factor giving duration closest to ideal bowling video length
                        best = min(candidates, key=lambda c: c[3])
                        print(f"Slo-mo: selected {best[1]}fps ({best[0]}x) → {best[2]:.1f}s real duration")
                        return round(best[0], 1)
                        
        return 1.0
    except Exception as e:
        print(f"Slo-mo detection failed: {e}")
        return 1.0


def postprocess_ball(output, scale, w, h, conf_thresh=0.15):
    """Extract ball detections — auto-detects custom 1-class model vs COCO 80-class."""
    predictions = output[0]
    if len(predictions.shape) == 3:
        predictions = predictions[0].T

    if len(predictions) == 0:
        return []

    # Auto-detect model format from output shape
    # YOLOv8/v11 detection: [x, y, w, h, class_scores...]
    # COCO 80-class: 84 columns (4 box + 80 classes)
    # Custom 1-class: 5 columns (4 box + 1 class)
    num_cols = predictions.shape[1] if len(predictions.shape) == 2 else 0
    boxes = predictions[:, :4]

    if num_cols <= 6:
        # Custom model: 1-2 classes — ball is class 0
        ball_scores = predictions[:, 4]
        print(f"Ball detection: custom model format ({num_cols} cols, 1-class)")
    elif num_cols >= 84:
        # COCO format: 80 classes, sports ball = class 32
        SPORTS_BALL_CLASS = 32
        class_scores = predictions[:, 4:]
        if class_scores.shape[1] <= SPORTS_BALL_CLASS:
            return []
        ball_scores = class_scores[:, SPORTS_BALL_CLASS]
        print(f"Ball detection: COCO format ({num_cols} cols, using class 32)")
    else:
        # Unknown format — try treating last column as score
        ball_scores = predictions[:, 4] if num_cols > 4 else predictions[:, -1]
        print(f"Ball detection: unknown format ({num_cols} cols, using col 4)")

    mask = ball_scores > conf_thresh
    ball_boxes = boxes[mask]
    ball_confs = ball_scores[mask]

    if len(ball_boxes) == 0:
        return []

    detections = []
    for i in range(len(ball_boxes)):
        cx, cy, bw, bh = ball_boxes[i]
        detections.append({
            "x": round(float(cx / scale), 1),
            "y": round(float(cy / scale), 1),
            "w": round(float(bw / scale), 1),
            "h": round(float(bh / scale), 1),
            "confidence": round(float(ball_confs[i]), 3),
            "nx": round(float(cx / scale / w), 4),
            "ny": round(float(cy / scale / h), 4),
        })

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections[:3]


def process_ball_tracking(job_id):
    """Detect cricket ball using HSV color detection + contour analysis + trajectory fitting."""
    job = jobs[job_id]
    tmp_path = job["tmp_path"]
    ball_color = job.get("ball_color", "red")

    try:
        slomo_factor = detect_slomo(tmp_path)
        cap = cv2.VideoCapture(tmp_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Analyze every frame for slo-mo, every 2nd for 30fps
        # For Roboflow API: use larger skip to avoid excessive API calls
        use_ml = ball_session is not None
        use_roboflow = bool(ROBOFLOW_API_KEY)
        if use_roboflow:
            # Target ~30 frames per second of real time for good trajectory resolution
            skip = max(1, int(video_fps * slomo_factor / 30))
        else:
            skip = max(1, int(video_fps / 60))
        total_analysis = total_frames // skip
        method = "roboflow_api" if use_roboflow else ("yolov11_local" if use_ml else "hsv_color")
        job["video_info"] = {
            "width": width, "height": height,
            "fps": round(video_fps, 2),
            "total_frames": total_frames,
            "slomo_factor": slomo_factor,
            "real_duration": round(total_frames / video_fps / slomo_factor, 3),
            "method": method,
        }
        job["total_frames"] = total_analysis
        job["status"] = "processing"

        raw_candidates = []
        frames_done = 0

        # ═══ PRIMARY: Roboflow hosted API (GPU-accelerated) ═══
        if use_roboflow:
            print(f"Ball job {job_id}: testing Roboflow API ({ROBOFLOW_MODEL_ID})...")
            # Test on first frame to verify API is working
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            ret, test_frame = cap.read()
            if ret:
                test_result = detect_ball_roboflow(test_frame, conf_thresh=0.05)
                # Also do a raw API call to see ALL classes being detected
                _, test_enc = cv2.imencode('.jpg', test_frame)
                try:
                    test_resp = requests.post(ROBOFLOW_URL, params={"api_key": ROBOFLOW_API_KEY, "confidence": 5}, files={"file": ("test.jpg", test_enc.tobytes(), "image/jpeg")}, timeout=15)
                    if test_resp.status_code == 200:
                        all_preds = test_resp.json().get("predictions", [])
                        from collections import Counter
                        classes = Counter(p.get("class", "?") for p in all_preds)
                        print(f"Ball job {job_id}: Roboflow test — ALL classes: {dict(classes)}")
                        ball_only = [p for p in all_preds if p.get("class", "").lower() == "ball"]
                        print(f"Ball job {job_id}: Roboflow test — {len(ball_only)} ball detections, {len(all_preds) - len(ball_only)} non-ball filtered out")
                except Exception as te:
                    print(f"Ball job {job_id}: test frame debug failed: {te}")
                print(f"Ball job {job_id}: Roboflow test — {len(test_result)} detections on middle frame")
                if test_result is None:
                    # API is broken, fall back
                    print(f"Ball job {job_id}: Roboflow API failed, falling back to HSV")
                    use_roboflow = False
                    method = "hsv_color"
                    job["video_info"]["method"] = method

        if use_roboflow:
            # PASS 0: BACKGROUND FALSE POSITIVE DETECTION
            # Run detection on first 3 frames (before bowling starts).
            # Any "ball" detections are false positives (stumps, net marks, pitch marks).
            # Record their positions and exclude them from all subsequent frames.
            false_positive_positions = []
            for bg_frame_idx in [0, 4, 8]:
                if bg_frame_idx < total_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, bg_frame_idx)
                    ret, frame = cap.read()
                    if ret:
                        bg_dets = detect_ball_roboflow(frame, conf_thresh=0.05)
                        for det in bg_dets:
                            false_positive_positions.append((round(det["nx"], 2), round(det["ny"], 2)))
            
            if false_positive_positions:
                print(f"Ball job {job_id}: Pass 0 — found {len(false_positive_positions)} false positive positions (stumps/net marks):")
                for fp in false_positive_positions:
                    print(f"  FP at ({fp[0]:.2f}, {fp[1]:.2f})")
            
            def is_false_positive(x, y, fp_list, tolerance=0.04):
                """Check if detection is near a known false positive position."""
                for fpx, fpy in fp_list:
                    if abs(x - fpx) < tolerance and abs(y - fpy) < tolerance:
                        return True
                return False
            
            # THREE-PASS APPROACH:
            # Pass 0: Background detection (done above)
            # Pass 1: Coarse scan (skip=8) to find rough time window where ball appears
            # Pass 2: Dense scan (skip=2) in that window for accurate trajectory
            coarse_skip = max(1, int(video_fps * slomo_factor / 30))
            total_analysis = total_frames // coarse_skip
            print(f"Ball job {job_id}: Pass 1 — coarse scan, {total_analysis} frames (skip={coarse_skip})")
            
            coarse_candidates = []
            target_frame = 0
            while target_frame < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                ret, frame = cap.read()
                if not ret:
                    break
                time_s = target_frame / video_fps / slomo_factor
                detections = detect_ball_roboflow(frame, conf_thresh=0.08)
                if detections:
                    for det in detections:
                        nx, ny = round(det["nx"], 4), round(det["ny"], 4)
                        if not is_false_positive(nx, ny, false_positive_positions):
                            coarse_candidates.append({
                                "frame": target_frame, "time": round(time_s, 4),
                                "x": nx, "y": ny,
                                "conf": det["confidence"],
                            })
                frames_done += 1
                job["frames_done"] = frames_done
                job["progress"] = round((frames_done / max(total_analysis, 1)) * 40)
                frames_done += 1
                job["frames_done"] = frames_done
                job["progress"] = round((frames_done / max(total_analysis, 1)) * 40)
                target_frame += coarse_skip
            
            print(f"Ball job {job_id}: Pass 1 found {len(coarse_candidates)} raw detections")
            
            # Remove static clusters from coarse pass
            from collections import defaultdict
            grid = defaultdict(list)
            for i, c in enumerate(coarse_candidates):
                gx = round(c["x"] / 0.03)
                gy = round(c["y"] / 0.03)
                grid[(gx, gy)].append(i)
            
            static_indices = set()
            threshold = max(3, len(coarse_candidates) * 0.3)
            for cell, indices in grid.items():
                if len(indices) >= threshold:
                    print(f"Ball job {job_id}: STATIC cluster at ({cell[0]*0.03:.2f}, {cell[1]*0.03:.2f}) — {len(indices)} detections")
                    static_indices.update(indices)
            
            moving = [c for i, c in enumerate(coarse_candidates) if i not in static_indices]
            print(f"Ball job {job_id}: After static removal: {len(moving)} moving detections")
            
            # Pass 2: Dense scan around any moving detections
            if moving:
                # Find time window: expand around detected frames
                det_frames = sorted(set(c["frame"] for c in moving))
                min_frame = max(0, min(det_frames) - coarse_skip * 5)
                max_frame = min(total_frames, max(det_frames) + coarse_skip * 5)
                dense_frames = max_frame - min_frame
                print(f"Ball job {job_id}: Pass 2 — dense scan frames {min_frame}-{max_frame} ({dense_frames} frames, skip=2)")
                
                dense_skip = 2  # analyze every 2nd frame in the window
                target_frame = min_frame
                while target_frame < max_frame:
                    # Skip frames already analyzed in coarse pass
                    if target_frame % coarse_skip != 0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                        ret, frame = cap.read()
                        if ret:
                            time_s = target_frame / video_fps / slomo_factor
                            detections = detect_ball_roboflow(frame, conf_thresh=0.08)
                            if detections:
                                for det in detections:
                                    nx, ny = round(det["nx"], 4), round(det["ny"], 4)
                                    if not is_false_positive(nx, ny, false_positive_positions):
                                        moving.append({
                                            "frame": target_frame, "time": round(time_s, 4),
                                            "x": nx, "y": ny,
                                            "conf": det["confidence"],
                                        })
                        frames_done += 1
                        job["frames_done"] = frames_done
                    target_frame += dense_skip
                
                # Remove static from dense pass too
                grid2 = defaultdict(list)
                for i, c in enumerate(moving):
                    gx = round(c["x"] / 0.03)
                    gy = round(c["y"] / 0.03)
                    grid2[(gx, gy)].append(i)
                static2 = set()
                threshold2 = max(3, len(moving) * 0.3)
                for cell, indices in grid2.items():
                    if len(indices) >= threshold2:
                        static2.update(indices)
                if static2:
                    moving = [c for i, c in enumerate(moving) if i not in static2]
                    print(f"Ball job {job_id}: Dense pass static removal: {len(static2)} removed, {len(moving)} remaining")
                
                raw_candidates = sorted(moving, key=lambda c: c["frame"])
                print(f"Ball job {job_id}: Total moving detections after both passes: {len(raw_candidates)}")
            else:
                print(f"Ball job {job_id}: No moving detections in coarse pass")
                raw_candidates = []
            
            cap.release()
            
            if raw_candidates:
                for i, c in enumerate(raw_candidates[:10]):
                    print(f"Ball job {job_id}: detection {i}: frame={c['frame']} x={c['x']:.3f} y={c['y']:.3f} conf={c['conf']:.3f}")

        # ═══ SECONDARY: Local ONNX model ═══
        elif use_ml:
            print(f"Ball job {job_id}: using ML detection (custom YOLOv11)")
            input_name = ball_session.get_inputs()[0].name
            target_frame = 0
            while target_frame < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                ret, frame = cap.read()
                if not ret:
                    break
                time_s = target_frame / video_fps / slomo_factor
                blob, scale = preprocess(frame)
                outputs = ball_session.run(None, {input_name: blob})
                detections = postprocess_ball(outputs, scale, width, height)
                if detections:
                    best = detections[0]
                    raw_candidates.append({
                        "frame": target_frame, "time": round(time_s, 4),
                        "x": best["nx"], "y": best["ny"],
                        "conf": best["confidence"],
                    })
                frames_done += 1
                job["frames_done"] = frames_done
                job["progress"] = round((frames_done / max(total_analysis, 1)) * 80)
                target_frame += skip

            cap.release()
            print(f"Ball job {job_id}: ML detection found {len(raw_candidates)} candidates in {frames_done} frames")

        # ═══ FALLBACK: HSV color detection ═══
        else:
            print(f"Ball job {job_id}: using HSV color fallback")

            # STEP 1: Build background model
            bg_indices = np.linspace(0, total_frames - 1, 15, dtype=int)
            bg_frames = []
            for idx in bg_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret:
                    bg_frames.append(frame.astype(np.float32))
            bg_model = np.median(np.array(bg_frames), axis=0).astype(np.uint8) if len(bg_frames) >= 5 else None
            bg_hsv = cv2.cvtColor(bg_model, cv2.COLOR_BGR2HSV) if bg_model is not None else None
            print(f"Ball job {job_id}: background model from {len(bg_frames)} frames")

            # ═══ STEP 2: Detect ball candidates in each frame ═══
            # HSV ranges based on selected ball color
            hsv_ranges = {
                'red': [
                    (np.array([0, 70, 50]), np.array([12, 255, 255])),
                    (np.array([165, 70, 50]), np.array([180, 255, 255])),
                ],
                'pink': [
                    (np.array([140, 30, 100]), np.array([175, 200, 255])),
                    (np.array([0, 40, 150]), np.array([10, 180, 255])),  # pinkish-red
                ],
                'white': [
                    (np.array([0, 0, 200]), np.array([180, 50, 255])),
                ],
            }
            color_ranges = hsv_ranges.get(ball_color, hsv_ranges['red'])
            print(f"Ball job {job_id}: detecting {ball_color} ball with {len(color_ranges)} HSV ranges")

            # Pitch-only mask: center strip of frame
            pitch_mask = np.zeros((height, width), dtype=np.uint8)
            px1, px2 = int(width * 0.25), int(width * 0.75)
            pitch_mask[:, px1:px2] = 255

            # Ball size constraints (in pixels)
            min_ball_area = max(3, int(width * height * 0.00002))  # Tiny at far end
            max_ball_area = int(width * height * 0.005)  # Bigger close to camera

            raw_candidates = []
            frames_done = 0

            target_frame = 0
            while target_frame < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                ret, frame = cap.read()
                if not ret:
                    break

                time_s = target_frame / video_fps / slomo_factor
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # Color mask for selected ball color
                color_mask = np.zeros((height, width), dtype=np.uint8)
                for lower, upper in color_ranges:
                    color_mask = cv2.bitwise_or(color_mask, cv2.inRange(hsv, lower, upper))

                # Subtract background color to reduce static colored objects
                if bg_hsv is not None:
                    bg_color = np.zeros((height, width), dtype=np.uint8)
                    for lower, upper in color_ranges:
                        bg_color = cv2.bitwise_or(bg_color, cv2.inRange(bg_hsv, lower, upper))
                    bg_color = cv2.dilate(bg_color, np.ones((5, 5), np.uint8), iterations=2)
                    color_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(bg_color))

                # Also detect via frame difference from background (catches any color ball)
                if bg_model is not None:
                    diff = cv2.absdiff(frame, bg_model)
                    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                    _, motion_mask = cv2.threshold(diff_gray, 35, 255, cv2.THRESH_BINARY)
                    # Combine: pixels that are EITHER ball-colored OR moving (but must be on pitch)
                    combined = cv2.bitwise_or(color_mask, motion_mask)
                else:
                    combined = color_mask

                # Apply pitch mask
                combined = cv2.bitwise_and(combined, pitch_mask)

                # Morphological cleanup
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                combined = cv2.erode(combined, kernel, iterations=1)
                combined = cv2.dilate(combined, kernel, iterations=2)

                # Find contours
                contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                frame_candidates = []
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < min_ball_area or area > max_ball_area:
                        continue
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    aspect = max(bw, bh) / max(min(bw, bh), 1)
                    if aspect > 3.0:  # Ball is roundish
                        continue
                    # Circularity check
                    perimeter = cv2.arcLength(cnt, True)
                    circularity = 4 * np.pi * area / max(perimeter * perimeter, 1)
                    cx = (x + bw / 2) / width
                    cy = (y + bh / 2) / height
                    # Score by circularity and color match strength
                    color_score = cv2.mean(color_mask[y:y+bh, x:x+bw])[0] / 255
                    score = circularity * 0.5 + color_score * 0.5
                    frame_candidates.append({
                        "x": round(cx, 4), "y": round(cy, 4),
                        "area": area, "score": round(score, 3),
                        "circ": round(circularity, 2),
                    })

                # Keep best candidate per frame
                if frame_candidates:
                    frame_candidates.sort(key=lambda c: c["score"], reverse=True)
                    best = frame_candidates[0]
                    raw_candidates.append({
                        "frame": target_frame, "time": round(time_s, 4),
                        "x": best["x"], "y": best["y"],
                        "conf": best["score"],
                    })

                frames_done += 1
                job["frames_done"] = frames_done
                job["progress"] = round((frames_done / max(total_analysis, 1)) * 80)
                target_frame += skip

            cap.release()
            print(f"Ball job {job_id}: {len(raw_candidates)} raw HSV candidates from {frames_done} frames")

        # ═══ Trajectory filtering ═══
        result = filter_ball_trajectory(raw_candidates, width, height)

        job["ball_positions"] = result.get("all_positions", [])
        job["deliveries"] = result.get("deliveries", [])
        job["video_info"]["frames_analyzed"] = frames_done
        job["video_info"]["raw_candidates"] = len(raw_candidates)
        job["video_info"]["deliveries_found"] = len(result.get("deliveries", []))
        job["status"] = "complete"
        job["progress"] = 100
        print(f"Ball job {job_id}: {len(result.get('deliveries', []))} deliveries from {len(raw_candidates)} candidates")

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        print(f"Ball job {job_id}: error — {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass


def filter_ball_trajectory(candidates, width, height):
    """Filter raw candidates to find ball trajectories.
    Returns ALL valid deliveries, scored and sorted by quality."""
    if len(candidates) < 2:
        return {"deliveries": [], "all_positions": candidates}

    # Log candidate distribution for debugging
    times = [c["time"] for c in candidates]
    print(f"Trajectory filter: {len(candidates)} candidates, time range {min(times):.3f}s - {max(times):.3f}s")
    if len(candidates) >= 2:
        gaps = [times[i+1] - times[i] for i in range(len(times)-1)]
        print(f"  Time gaps: min={min(gaps):.3f}s, max={max(gaps):.3f}s, median={sorted(gaps)[len(gaps)//2]:.3f}s")
        ys = [c["y"] for c in candidates]
        print(f"  Y range: {min(ys):.3f} - {max(ys):.3f} (span {max(ys)-min(ys):.3f})")

    # Adaptive gap threshold: use 3× the median gap between detections
    if len(candidates) >= 3:
        gaps = sorted([times[i+1] - times[i] for i in range(len(times)-1)])
        median_gap = gaps[len(gaps)//2]
        gap_threshold = max(0.3, min(2.0, median_gap * 3))
    else:
        gap_threshold = 1.0
    print(f"  Gap threshold: {gap_threshold:.3f}s")

    # Group candidates by time into potential deliveries
    deliveries = []
    current = [candidates[0]]
    for i in range(1, len(candidates)):
        time_gap = candidates[i]["time"] - candidates[i - 1]["time"]
        if time_gap > gap_threshold:
            if len(current) >= 2:
                deliveries.append(current)
            current = []
        current.append(candidates[i])
    if len(current) >= 2:
        deliveries.append(current)

    print(f"  Grouped into {len(deliveries)} potential deliveries: {[len(d) for d in deliveries]}")

    if not deliveries:
        # If grouping failed, treat ALL candidates as one delivery
        print(f"  No groups found, treating all {len(candidates)} as one delivery")
        deliveries = [candidates]

    # Score and filter each delivery
    valid_deliveries = []
    for idx, delivery in enumerate(deliveries):
        if len(delivery) < 2:
            continue

        # ═══ RELEASE POINT DETECTION ═══
        # From behind stumps: ball-in-hand moves UPWARD (Y decreasing) as arm goes over.
        # After release, ball moves DOWNWARD (Y increasing) towards camera.
        # Find the direction change point and trim everything before it.
        if len(delivery) >= 3:
            y_vals_raw = [p["y"] for p in delivery]
            # Find the minimum Y point (highest in frame = top of bowling arc)
            min_y_idx = y_vals_raw.index(min(y_vals_raw))
            # If the minimum Y is not at the start, there are ball-in-hand detections
            if min_y_idx > 0 and min_y_idx < len(delivery) - 1:
                # Trim everything before the release (min Y point)
                trimmed = delivery[min_y_idx:]
                if len(trimmed) >= 2:
                    print(f"  Group {idx}: release detected at idx {min_y_idx} (y={y_vals_raw[min_y_idx]:.3f}), trimmed {min_y_idx} ball-in-hand detections")
                    delivery = trimmed

        y_vals = [p["y"] for p in delivery]
        x_vals = [p["x"] for p in delivery]

        y_increasing = sum(1 for i in range(1, len(y_vals)) if y_vals[i] > y_vals[i-1])
        y_ratio = y_increasing / max(len(y_vals) - 1, 1)
        x_range = max(x_vals) - min(x_vals)
        y_span = max(y_vals) - min(y_vals)

        score = y_span * max(y_ratio, 0.3) * len(delivery) / max(x_range * 10, 0.1)
        transit = delivery[-1]["time"] - delivery[0]["time"]

        print(f"  Group {idx}: {len(delivery)} pts, y_span={y_span:.3f}, y_ratio={y_ratio:.2f}, x_range={x_range:.3f}, transit={transit:.3f}s, score={score:.2f}")

        # Relaxed criteria: just need some vertical movement and reasonable transit time
        if y_span > 0.02 and transit > 0.02 and transit < 5.0:
            valid_deliveries.append({
                "positions": delivery,
                "score": round(score, 2),
                "transit_time": round(transit, 4),
                "start_time": delivery[0]["time"],
                "end_time": delivery[-1]["time"],
                "detections": len(delivery),
            })

    # If still nothing, just return the largest group as-is
    # BUT only if there's actual movement (not a static object like stumps)
    if not valid_deliveries and deliveries:
        largest = max(deliveries, key=len)
        transit = largest[-1]["time"] - largest[0]["time"]
        y_vals_lg = [p["y"] for p in largest]
        x_vals_lg = [p["x"] for p in largest]
        y_span_lg = max(y_vals_lg) - min(y_vals_lg)
        x_span_lg = max(x_vals_lg) - min(x_vals_lg)
        total_movement = y_span_lg + x_span_lg
        if transit > 0 and len(largest) >= 2 and total_movement > 0.03:
            print(f"  Fallback: using largest group ({len(largest)} pts, movement={total_movement:.3f})")
            valid_deliveries.append({
                "positions": largest,
                "score": 1.0,
                "transit_time": round(transit, 4),
                "start_time": largest[0]["time"],
                "end_time": largest[-1]["time"],
                "detections": len(largest),
            })
        else:
            print(f"  Fallback rejected: movement={total_movement:.3f} too small (likely static object like stumps)")

    valid_deliveries.sort(key=lambda d: d["score"], reverse=True)
    print(f"  Result: {len(valid_deliveries)} valid deliveries")
    return {"deliveries": valid_deliveries, "all_positions": candidates}


def process_video(job_id):
    """Background thread: process video frames and update job progress."""
    job = jobs[job_id]
    tmp_path = job["tmp_path"]
    fps_hint = job["fps"]

    try:
        # Use user's Video Type selection as slo-mo override
        # fps_hint = 30 (normal), 120 (slo-mo 4x), 240 (slo-mo 8x)
        override_factor = None
        if fps_hint and fps_hint > 30:
            # User explicitly selected slo-mo mode
            # Video file is 30fps display, native capture was fps_hint
            override_factor = fps_hint / 30
            print(f"User selected {fps_hint}fps → override slo-mo factor to {override_factor}x")
        
        slomo_factor = detect_slomo(tmp_path, override_factor=override_factor)
        print(f"Slo-mo detection: factor={slomo_factor}")

        cap = cv2.VideoCapture(tmp_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or fps_hint
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Skip frames based on video fps and slo-mo
        # skip=2: 164 frames for 240fps slo-mo (8.3ms gaps, ~8 min processing)
        # Balances accuracy vs pose jitter noise at 70px bowler resolution
        target_real_fps = 120  # skip=2 for 240fps slo-mo
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
        
        print(f"Frame sampling: video={video_fps}fps, slomo={slomo_factor}x, skip={skip} → {video_fps/skip:.1f} video fps = {video_fps/skip*slomo_factor:.0f} real fps")

        analysis_fps = video_fps / skip
        total_analysis = total_frames // skip

        job["video_info"] = {
            "width": width,
            "height": height,
            "fps": round(video_fps, 2),
            "total_frames": total_frames,
            "duration": round(total_frames / video_fps, 3) if video_fps > 0 else 0,
            "analysis_fps": round(analysis_fps, 1),
            "effective_fps": round(analysis_fps * slomo_factor, 1),  # Real-time fps after slo-mo correction
            "slomo_factor": slomo_factor,
            "real_duration": round(total_frames / video_fps / slomo_factor, 3) if video_fps > 0 else 0,
        }
        job["total_frames"] = total_analysis
        job["status"] = "processing"

        frames = []
        frame_idx = 0
        frames_done = 0
        input_name = session.get_inputs()[0].name

        print(f"Dual detection: full-frame + crop-zoom for each frame")
        print(f"  Full-frame: {width}×{height} → scale={min(640/width,640/height):.3f} (for events, strides, distances)")
        print(f"  Crop-zoom: bowler crop at ~3x resolution (for angles)")

        # Use seeking for high-fps videos (much faster than read-skip)
        target_frame = 0
        while target_frame < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            if not ret:
                break

            time_s = target_frame / video_fps / slomo_factor
            
            # Pass 1: Full-frame (stable coordinates for events + distances)
            blob, scale = preprocess(frame)
            outputs = session.run(None, {input_name: blob})
            landmarks, confidence = postprocess(outputs, scale)
            
            # Pass 2: Crop-zoom (better joint accuracy for angles)
            cropped_landmarks = None
            if landmarks:
                xs = [lm["x"] for lm in landmarks if lm is not None]
                ys = [lm["y"] for lm in landmarks if lm is not None]
                if len(xs) >= 4:
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    box_w = max_x - min_x
                    box_h = max_y - min_y
                    pad_x = box_w * 0.5
                    pad_y = box_h * 0.4
                    cx1 = max(0, int(min_x - pad_x))
                    cy1 = max(0, int(min_y - pad_y))
                    cx2 = min(width, int(max_x + pad_x))
                    cy2 = min(height, int(max_y + pad_y))
                    crop = frame[cy1:cy2, cx1:cx2]
                    blob2, scale2 = preprocess(crop)
                    outputs2 = session.run(None, {input_name: blob2})
                    lm2, conf2 = postprocess(outputs2, scale2)
                    if lm2 and conf2 > confidence * 0.5:
                        # Map crop coordinates back to full frame
                        cropped_landmarks = [None] * 33
                        for idx, lm in enumerate(lm2):
                            if lm is not None:
                                cropped_landmarks[idx] = {
                                    "x": round(lm["x"] + cx1, 1),
                                    "y": round(lm["y"] + cy1, 1),
                                    "z": lm["z"],
                                    "visibility": lm["visibility"],
                                }

            frames.append({
                "frame": target_frame,
                "time": round(time_s, 4),
                "keypoints": landmarks,
                "cropped_keypoints": cropped_landmarks,
                "confidence": confidence,
            })

            frames_done += 1
            job["frames_done"] = frames_done
            job["progress"] = round((frames_done / max(total_analysis, 1)) * 100)
            target_frame += skip

        cap.release()

        # ═══ DIAGNOSTIC OUTPUT ═══
        # Trace actual data at each pipeline step
        valid_frames = [f for f in frames if f["keypoints"] is not None]
        print(f"\n{'='*60}")
        print(f"PIPELINE DIAGNOSTIC — Job {job_id}")
        print(f"{'='*60}")
        print(f"Step 1 - Video: {width}×{height}, {video_fps}fps, {total_frames} frames, slomo={slomo_factor}x")
        print(f"Step 2 - Sampling: skip={skip}, analyzed={len(frames)}, valid={len(valid_frames)} ({100*len(valid_frames)//max(len(frames),1)}%)")
        scale_diag = min(640/width, 640/height)
        print(f"Step 3 - Two-pass crop detection: full frame scale={scale_diag:.3f}, crop ~3-4x better")
        
        # Step 4: Sample actual keypoint values
        if len(valid_frames) >= 5:
            sample_indices = [0, len(valid_frames)//4, len(valid_frames)//2, 3*len(valid_frames)//4, len(valid_frames)-1]
            print(f"\nStep 4 - Sample keypoints (5 frames across video):")
            for si in sample_indices:
                f = valid_frames[si]
                kp = f["keypoints"]
                # Left hip (23), Right hip (24), Left ankle (27), Right ankle (28), Right wrist (16)
                hip_l = kp[23] if kp[23] else None
                hip_r = kp[24] if kp[24] else None
                ank_l = kp[27] if kp[27] else None
                ank_r = kp[28] if kp[28] else None
                wrist_r = kp[16] if kp[16] else None
                wrist_l = kp[15] if kp[15] else None
                hip_mid = None
                if hip_l and hip_r:
                    hip_mid = ((hip_l["x"]+hip_r["x"])/2, (hip_l["y"]+hip_r["y"])/2)
                
                print(f"  Frame {f['frame']} (t={f['time']:.3f}s, conf={f['confidence']:.2f}):")
                if hip_mid:
                    print(f"    Hip mid: ({hip_mid[0]:.1f}, {hip_mid[1]:.1f})")
                if hip_l:
                    print(f"    L.Hip: ({hip_l['x']:.1f}, {hip_l['y']:.1f}) vis={hip_l['visibility']:.2f}")
                if ank_l:
                    print(f"    L.Ankle: ({ank_l['x']:.1f}, {ank_l['y']:.1f}) vis={ank_l['visibility']:.2f}")
                if ank_r:
                    print(f"    R.Ankle: ({ank_r['x']:.1f}, {ank_r['y']:.1f}) vis={ank_r['visibility']:.2f}")
                if wrist_r:
                    print(f"    R.Wrist: ({wrist_r['x']:.1f}, {wrist_r['y']:.1f}) vis={wrist_r['visibility']:.2f}")
            
            # Step 5: Frame-to-frame hip displacement (first 10 consecutive valid frames)
            print(f"\nStep 5 - Hip displacement between consecutive frames:")
            for i in range(1, min(11, len(valid_frames))):
                prev_f = valid_frames[i-1]
                curr_f = valid_frames[i]
                ph = prev_f["keypoints"]
                ch = curr_f["keypoints"]
                if ph[23] and ph[24] and ch[23] and ch[24]:
                    px = (ph[23]["x"]+ph[24]["x"])/2
                    py = (ph[23]["y"]+ph[24]["y"])/2
                    cx = (ch[23]["x"]+ch[24]["x"])/2
                    cy = (ch[23]["y"]+ch[24]["y"])/2
                    dx = cx - px
                    dy = cy - py
                    dist = (dx**2 + dy**2)**0.5
                    dt = curr_f["time"] - prev_f["time"]
                    speed_px_s = dist / dt if dt > 0 else 0
                    print(f"  Frame {prev_f['frame']}→{curr_f['frame']}: dx={dx:.1f} dy={dy:.1f} dist={dist:.1f}px dt={dt*1000:.1f}ms → {speed_px_s:.0f}px/s")
            
            # Step 6: Ankle positions around likely delivery (max stride)
            max_stride_px = 0
            max_stride_idx = 0
            for i, f in enumerate(valid_frames):
                kp = f["keypoints"]
                if kp[27] and kp[28] and kp[27]["visibility"] > 0.2 and kp[28]["visibility"] > 0.2:
                    stride = abs(kp[27]["x"] - kp[28]["x"])
                    if stride > max_stride_px:
                        max_stride_px = stride
                        max_stride_idx = i
            
            print(f"\nStep 6 - Max stride detection:")
            print(f"  Max ankle X separation: {max_stride_px:.1f}px at frame {valid_frames[max_stride_idx]['frame']}")
            # Show ankle positions around max stride
            for di in range(-3, 4):
                idx = max_stride_idx + di
                if 0 <= idx < len(valid_frames):
                    f = valid_frames[idx]
                    kp = f["keypoints"]
                    la = kp[27]
                    ra = kp[28]
                    sep = abs(la["x"] - ra["x"]) if la and ra else 0
                    print(f"  [{'+' if di>0 else ''}{di}] Frame {f['frame']} t={f['time']:.3f}s: L.Ank=({la['x']:.0f},{la['y']:.0f}) R.Ank=({ra['x']:.0f},{ra['y']:.0f}) sep={sep:.0f}px" if la and ra else f"  [{'+' if di>0 else ''}{di}] Frame {f['frame']}: ankle data missing")

            # Step 7: Bowler apparent height across video
            print(f"\nStep 7 - Bowler height across video:")
            for si in sample_indices:
                f = valid_frames[si]
                kp = f["keypoints"]
                nose = kp[0]
                ank_l = kp[27]
                ank_r = kp[28]
                if nose and (ank_l or ank_r):
                    ank_y = max(ank_l["y"] if ank_l else 0, ank_r["y"] if ank_r else 0)
                    h_px = ank_y - nose["y"]
                    print(f"  Frame {f['frame']}: nose_y={nose['y']:.0f} ank_y={ank_y:.0f} → height={h_px:.0f}px (vis={nose['visibility']:.2f})")
                else:
                    print(f"  Frame {f['frame']}: missing nose or ankle")
        
        print(f"{'='*60}\n")

        # Include diagnostics in response for frontend logging
        diag_scale = round(min(640/width, 640/height), 3)
        diag_input = f"{int(width*diag_scale)}x{int(height*diag_scale)}"
        diag = {
            "skip": skip,
            "scale": diag_scale,
            "model_input": diag_input,
            "valid_frames": len(valid_frames),
            "slomo_factor": slomo_factor,
            "real_fps": round(video_fps / skip * slomo_factor, 1) if slomo_factor > 1 else round(video_fps / skip, 1),
            "time_gap_ms": round(skip / video_fps / slomo_factor * 1000, 1) if slomo_factor > 1 else round(skip / video_fps * 1000, 1),
        }
        job["video_info"]["diagnostics"] = diag

        job["video_info"]["frames_analyzed"] = len(frames)
        job["frames"] = frames
        job["status"] = "complete"
        job["progress"] = 100
        print(f"Job {job_id}: complete — {len(frames)} frames analyzed")

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        print(f"Job {job_id}: error — {e}")
    finally:
        # Clean up temp file
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass


@app.on_event("startup")
async def startup():
    global session, ball_session
    download_model()
    print("Loading ONNX pose model...")
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    print(f"Pose model loaded! Input: {session.get_inputs()[0].shape}")

    # Load custom cricket ball detection model if available
    if os.path.exists(BALL_MODEL_PATH) and os.path.getsize(BALL_MODEL_PATH) > 100000:
        try:
            ball_session = ort.InferenceSession(BALL_MODEL_PATH, providers=['CPUExecutionProvider'])
            print(f"Cricket ball model loaded! Input: {ball_session.get_inputs()[0].shape}")
        except Exception as e:
            print(f"Cricket ball model failed to load: {e}")
            ball_session = None
    else:
        print("No cricket ball model found — ball detection will use HSV fallback")

    dummy = np.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
    session.run(None, {session.get_inputs()[0].name: dummy})
    if ball_session:
        try:
            ball_session.run(None, {ball_session.get_inputs()[0].name: dummy})
        except Exception as e:
            print(f"Ball model warmup failed: {e}")
    print(f"Models warmed up! Ball detection: {'Roboflow API (' + ROBOFLOW_MODEL_ID + ')' if ROBOFLOW_API_KEY else 'ML (custom YOLOv11)' if ball_session else 'HSV fallback'}")
    # Initialize Stripe (optional — works without it)
    if init_stripe():
        print("Stripe payment system ready")
    else:
        print("Stripe not configured — payment endpoints disabled")


@app.get("/health")
def health():
    ball_method = "roboflow_api" if ROBOFLOW_API_KEY else "local_onnx" if ball_session else "hsv_fallback"
    return {
        "status": "ok",
        "model_loaded": session is not None,
        "ball_detection": ball_method,
        "roboflow_model": ROBOFLOW_MODEL_ID if ROBOFLOW_API_KEY else None,
    }


@app.get("/")
def root():
    return {"service": "Cricket Analyze Pro - Pose API", "status": "running", "version": "2.0"}


@app.post("/analyze-standing")
async def analyze_standing(
    image: UploadFile = File(...),
    height_cm: float = Form(...),
):
    """
    Calibration endpoint: accepts a single standing (T-pose preferred) image
    and returns body-segment measurements in cm.

    Used by the optional Bowler Profile flow to derive a trustworthy
    pxPerCm reference plus real-world arm span, torso and leg lengths.
    These are then used by the analysis pipeline to do per-frame
    calibration from the bowler's torso length (most stable segment
    during a bowling action).
    """
    # Load pose model (same model used by the video pipeline)
    if session is None:
        return JSONResponse(status_code=503, content={"error": "pose model not ready"})

    # Read image
    content = await image.read()
    nparr = np.frombuffer(content, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse(status_code=400, content={"error": "could not decode image"})

    # Run pose detection (use the same cropped two-pass path for accuracy)
    landmarks, conf = detect_pose_cropped(frame, session, conf_thresh=0.2)
    if not landmarks:
        return JSONResponse(status_code=422, content={
            "error": "no person detected",
            "hint": "Use a clear, full-body photo with heels and crown visible"
        })

    # Extract keypoints we need (MediaPipe indices)
    def pt(i):
        lm = landmarks[i]
        if lm is None or lm.get("visibility", 0) < 0.15:
            return None
        return (lm["x"], lm["y"])

    nose = pt(0)
    l_shoulder, r_shoulder = pt(11), pt(12)
    l_wrist, r_wrist = pt(15), pt(16)
    l_hip, r_hip = pt(23), pt(24)
    l_ankle, r_ankle = pt(27), pt(28)

    missing = []
    if not nose: missing.append("head")
    if not (l_shoulder and r_shoulder): missing.append("shoulders")
    if not (l_wrist and r_wrist): missing.append("wrists")
    if not (l_hip and r_hip): missing.append("hips")
    if not (l_ankle and r_ankle): missing.append("ankles")
    if missing:
        return JSONResponse(status_code=422, content={
            "error": f"could not see: {', '.join(missing)}",
            "hint": "Stand facing the camera, arms out to the sides, full body in frame"
        })

    # Midpoints
    mid = lambda a, b: ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    shoulders_mid = mid(l_shoulder, r_shoulder)
    hips_mid = mid(l_hip, r_hip)
    ankles_mid = mid(l_ankle, r_ankle)

    # Pixel measurements
    def dist(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    # Heel-to-crown estimate: nose is top-of-nose, crown sits ~8% of stature above it.
    # We estimate crown by extending upward along the shoulders->nose vector.
    nose_to_shoulders = dist(nose, shoulders_mid)
    # Typical anthropometry: from shoulder mid to crown is ~1.3 × shoulder-to-nose
    crown_y = nose[1] - 0.25 * nose_to_shoulders  # add ~headtop correction above nose
    crown = (nose[0], crown_y)

    heel_to_crown_px = dist(ankles_mid, crown)
    arm_span_px = dist(l_wrist, r_wrist)
    torso_px = dist(shoulders_mid, hips_mid)
    leg_px = dist(hips_mid, ankles_mid)

    if heel_to_crown_px < 50:
        return JSONResponse(status_code=422, content={
            "error": "bowler appears too small in frame",
            "hint": "Stand further back or crop less aggressively — we need full body clearly visible"
        })

    # Reference px/cm from known height
    reference_px_per_cm = heel_to_crown_px / height_cm

    # Derive real-world cm for the other segments
    arm_span_cm = arm_span_px / reference_px_per_cm
    torso_cm = torso_px / reference_px_per_cm
    leg_cm = leg_px / reference_px_per_cm

    # Ratios (Pitchwolf-style)
    arm_height_ratio = arm_span_cm / height_cm
    torso_leg_ratio = torso_cm / leg_cm

    # Build classification (Pitchwolf gives "Core Driver", "Lever Dominant", etc.)
    if arm_height_ratio >= 1.03:
        build = "Lever Dominant"
        build_note = "Long arms relative to height — built for express pace through fast arm speed."
    elif arm_height_ratio >= 0.98:
        build = "Balanced Athlete"
        build_note = "Balanced proportions — adaptable to a range of bowling styles."
    elif torso_leg_ratio >= 0.80:
        build = "Power Base"
        build_note = "Long torso relative to legs — built for seam and swing over express pace."
    else:
        build = "Core Driver"
        build_note = "Built for trunk rotation power rather than long-lever speed."

    return JSONResponse(content={
        "status": "ok",
        "confidence": float(conf),
        "height_cm": height_cm,
        "measurements": {
            "arm_span_cm": round(arm_span_cm, 1),
            "torso_cm": round(torso_cm, 1),
            "leg_cm": round(leg_cm, 1),
            "arm_height_ratio": round(arm_height_ratio, 3),
            "torso_leg_ratio": round(torso_leg_ratio, 3),
        },
        "calibration": {
            "reference_px_per_cm": round(reference_px_per_cm, 4),
            "heel_to_crown_px": round(heel_to_crown_px, 1),
            "arm_span_px": round(arm_span_px, 1),
            "torso_px": round(torso_px, 1),
            "leg_px": round(leg_px, 1),
        },
        "profile": {
            "build": build,
            "note": build_note,
        },
    })


@app.post("/analyze")
async def analyze_video(
    video: UploadFile = File(...),
    fps: int = Form(30),
    height_cm: float = Form(0),
    bowling_arm: str = Form("right"),
):
    """Accept video upload and start background processing. Returns job_id immediately."""
    cleanup_jobs()  # Clean old jobs

    # Save uploaded video to temp file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        content = await video.read()
        tmp.write(content)
        tmp_path = tmp.name

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "frames_done": 0,
        "total_frames": 0,
        "created_at": time.time(),
        "tmp_path": tmp_path,
        "fps": fps,
        "height_cm": height_cm,
        "bowling_arm": bowling_arm,
    }

    # Start processing in background thread
    thread = threading.Thread(target=process_video, args=(job_id,), daemon=True)
    thread.start()

    return JSONResponse(content={"job_id": job_id, "status": "queued"})


@app.post("/upload-ball-model")
async def upload_ball_model(
    model: UploadFile = File(...),
    api_key: str = Form("cricket2026"),
):
    """Upload a custom ONNX ball detection model. Requires api_key for security."""
    global ball_session
    if api_key != "cricket2026":
        return JSONResponse(status_code=403, content={"error": "Invalid API key"})

    content = await model.read()
    if len(content) < 100000:
        return JSONResponse(status_code=400, content={"error": f"Model file too small ({len(content)} bytes)"})

    with open(BALL_MODEL_PATH, 'wb') as f:
        f.write(content)

    try:
        ball_session = ort.InferenceSession(BALL_MODEL_PATH, providers=['CPUExecutionProvider'])
        # Warm up
        dummy = np.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
        ball_session.run(None, {ball_session.get_inputs()[0].name: dummy})
        input_shape = ball_session.get_inputs()[0].shape
        output_shape = ball_session.get_outputs()[0].shape
        print(f"Custom ball model loaded! Input: {input_shape}, Output: {output_shape}")
        return {"status": "ok", "message": "Ball model loaded successfully", "input_shape": str(input_shape), "output_shape": str(output_shape), "size_mb": round(len(content) / 1e6, 1)}
    except Exception as e:
        ball_session = None
        if os.path.exists(BALL_MODEL_PATH):
            os.remove(BALL_MODEL_PATH)
        return JSONResponse(status_code=500, content={"error": f"Failed to load model: {str(e)}"})


@app.get("/status/{job_id}")
def get_status(job_id: str):
    """Poll this endpoint every 2 seconds to check processing progress."""
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "Job not found"})

    job = jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job.get("progress", 0),
        "frames_done": job.get("frames_done", 0),
        "total_frames": job.get("total_frames", 0),
        "error": job.get("error"),
    }


@app.get("/results/{job_id}")
def get_results(job_id: str):
    """Fetch full results once status is 'complete'."""
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "Job not found"})

    job = jobs[job_id]
    if job["status"] != "complete":
        return JSONResponse(status_code=202, content={
            "status": job["status"],
            "message": "Still processing" if job["status"] == "processing" else "Queued",
        })

    return JSONResponse(content={
        "video_info": job["video_info"],
        "frames": job["frames"],
    })


# ─── Ball Detection endpoints ─────────────────────────────────

@app.post("/detect-ball")
async def detect_ball_video(
    video: UploadFile = File(...),
    ball_color: str = Form("red"),
):
    """Upload video for ball tracking. Returns job_id — poll /ball-status/{job_id}."""
    cleanup_jobs()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        content = await video.read()
        tmp.write(content)
        tmp_path = tmp.name

    job_id = "ball-" + str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "frames_done": 0,
        "total_frames": 0,
        "created_at": time.time(),
        "tmp_path": tmp_path,
        "ball_color": ball_color,
    }

    thread = threading.Thread(target=process_ball_tracking, args=(job_id,), daemon=True)
    thread.start()

    return JSONResponse(content={"job_id": job_id, "status": "queued"})


@app.get("/ball-status/{job_id}")
def get_ball_status(job_id: str):
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    job = jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job.get("progress", 0),
        "frames_done": job.get("frames_done", 0),
        "total_frames": job.get("total_frames", 0),
        "error": job.get("error"),
    }


@app.get("/ball-results/{job_id}")
def get_ball_results(job_id: str):
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    job = jobs[job_id]
    if job["status"] != "complete":
        return JSONResponse(status_code=202, content={"status": job["status"]})

    return JSONResponse(content={
        "video_info": job.get("video_info", {}),
        "ball_positions": job.get("ball_positions", []),
        "deliveries": job.get("deliveries", []),
    })


# ─── Legacy sync endpoint (kept for backward compat) ─────────────

@app.post("/analyze-sync")
async def analyze_video_sync(
    video: UploadFile = File(...),
    fps: int = Form(30),
    height_cm: float = Form(0),
    bowling_arm: str = Form("right"),
):
    """Original synchronous endpoint — kept as fallback."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        content = await video.read()
        tmp.write(content)
        tmp_path = tmp.name
    try:
        cap = cv2.VideoCapture(tmp_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        skip = max(1, int(video_fps / 60))
        analysis_fps = video_fps / skip
        frames = []
        frame_idx = 0
        input_name = session.get_inputs()[0].name
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % skip != 0:
                frame_idx += 1
                continue
            time_s = frame_idx / video_fps
            blob, scale = preprocess(frame)
            outputs = session.run(None, {input_name: blob})
            landmarks, confidence = postprocess(outputs, scale)
            frames.append({
                "frame": frame_idx,
                "time": round(time_s, 4),
                "keypoints": landmarks,
                "confidence": confidence,
            })
            frame_idx += 1
        cap.release()
        return JSONResponse(content={
            "video_info": {
                "width": width, "height": height,
                "fps": round(video_fps, 2), "total_frames": total_frames,
                "duration": round(total_frames / video_fps, 3),
                "analysis_fps": round(analysis_fps, 1),
                "frames_analyzed": len(frames),
            },
            "frames": frames,
        })
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
