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
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1000000:
        print(f"Pose model already exists: {os.path.getsize(MODEL_PATH)/1e6:.1f} MB")
        return
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    print(f"Downloading pose model...")
    r = requests.get(MODEL_URL, allow_redirects=True, stream=True)
    r.raise_for_status()
    with open(MODEL_PATH, 'wb') as f:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
    size = os.path.getsize(MODEL_PATH)
    print(f"Pose model: {size/1e6:.1f} MB")
    if size < 1000000:
        raise RuntimeError(f"Model file too small ({size} bytes)")


def preprocess(frame, input_size=INPUT_SIZE):
    h, w = frame.shape[:2]
    scale = min(input_size / w, input_size / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh))
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    padded[:nh, :nw] = resized
    blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(blob, 0), scale


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


def detect_slomo(path):
    """Detect iPhone slo-mo videos and return the speedup factor needed."""
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

        # Check time_base for high native fps
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                time_base = stream.get("time_base", "")
                r_frame_rate = stream.get("r_frame_rate", "0/1")
                # Parse time_base denominator (e.g. "1/2400" → 2400)
                tb_denom = int(time_base.split("/")[1]) if "/" in time_base else 0
                # Parse display fps
                num, den = r_frame_rate.split("/") if "/" in r_frame_rate else ("0", "1")
                display_fps = int(num) / max(int(den), 1)

                # If time_base suggests much higher fps than display fps, it's slo-mo
                # e.g. time_base=1/2400 with display 60fps → native 240fps → 4x slowdown
                if tb_denom >= 240 and display_fps < 120 and full_rate == "0":
                    # Native fps is approximately time_base_denom / 10
                    native_fps = tb_denom / 10  # 2400/10=240, 1200/10=120
                    factor = native_fps / display_fps
                    if factor > 1.5:
                        return round(factor, 1)
        return 1.0
    except Exception as e:
        print(f"Slo-mo detection failed: {e}")
        return 1.0


def postprocess_ball(output, scale, w, h, conf_thresh=0.15):
    """Extract sports ball detections from YOLOv8 detection model output."""
    predictions = output[0]
    if len(predictions.shape) == 3:
        predictions = predictions[0].T  # (8400, 84) for COCO 80 classes

    if len(predictions) == 0:
        return []

    # YOLOv8 detection format: [x_center, y_center, w, h, class0_conf, class1_conf, ...]
    # COCO class 32 = sports ball
    SPORTS_BALL_CLASS = 32
    boxes = predictions[:, :4]
    class_scores = predictions[:, 4:]  # 80 classes

    if class_scores.shape[1] <= SPORTS_BALL_CLASS:
        return []

    ball_scores = class_scores[:, SPORTS_BALL_CLASS]
    mask = ball_scores > conf_thresh
    ball_boxes = boxes[mask]
    ball_confs = ball_scores[mask]

    if len(ball_boxes) == 0:
        return []

    detections = []
    for i in range(len(ball_boxes)):
        cx, cy, bw, bh = ball_boxes[i]
        # Scale back to original image coordinates
        detections.append({
            "x": round(float(cx / scale), 1),
            "y": round(float(cy / scale), 1),
            "w": round(float(bw / scale), 1),
            "h": round(float(bh / scale), 1),
            "confidence": round(float(ball_confs[i]), 3),
            # Normalized coordinates (0-1 range)
            "nx": round(float(cx / scale / w), 4),
            "ny": round(float(cy / scale / h), 4),
        })

    # Sort by confidence, return best detection
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections[:3]  # Top 3 candidates


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
        skip = max(1, int(video_fps / 60))
        total_analysis = total_frames // skip
        use_ml = ball_session is not None
        job["video_info"] = {
            "width": width, "height": height,
            "fps": round(video_fps, 2),
            "total_frames": total_frames,
            "slomo_factor": slomo_factor,
            "real_duration": round(total_frames / video_fps / slomo_factor, 3),
            "method": "yolov11_cricket" if use_ml else "hsv_color",
        }
        job["total_frames"] = total_analysis
        job["status"] = "processing"

        raw_candidates = []
        frames_done = 0

        # ═══ PRIMARY: ML ball detection (custom trained YOLOv11) ═══
        if use_ml:
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

        # ═══ STEP 3: Trajectory filtering ═══
        ball_positions = filter_ball_trajectory(raw_candidates, width, height)

        job["ball_positions"] = ball_positions
        job["video_info"]["frames_analyzed"] = frames_done
        job["video_info"]["raw_candidates"] = len(raw_candidates)
        job["video_info"]["filtered_positions"] = len(ball_positions)
        job["status"] = "complete"
        job["progress"] = 100
        print(f"Ball job {job_id}: {len(ball_positions)} trajectory points from {len(raw_candidates)} candidates")

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
    """Filter raw candidates to find the most likely ball trajectory.
    Ball should form a roughly straight/parabolic path from top to bottom."""
    if len(candidates) < 3:
        return candidates

    # Group candidates by time into potential deliveries
    # A delivery takes ~0.3-1.0 seconds in real time
    deliveries = []
    current = [candidates[0]]
    for i in range(1, len(candidates)):
        time_gap = candidates[i]["time"] - candidates[i - 1]["time"]
        if time_gap > 0.15:  # Gap > 150ms = new delivery or noise
            if len(current) >= 3:
                deliveries.append(current)
            current = []
        current.append(candidates[i])
    if len(current) >= 3:
        deliveries.append(current)

    if not deliveries:
        return []

    # Score each potential delivery by trajectory smoothness
    best_delivery = None
    best_score = -1

    for delivery in deliveries:
        if len(delivery) < 3:
            continue

        # Ball should move mostly downward (Y increases) and stay roughly centered
        y_vals = [p["y"] for p in delivery]
        x_vals = [p["x"] for p in delivery]

        # Check Y is generally increasing (ball moving towards camera)
        y_increasing = sum(1 for i in range(1, len(y_vals)) if y_vals[i] > y_vals[i-1])
        y_ratio = y_increasing / max(len(y_vals) - 1, 1)

        # Check X stays within reasonable lateral range (ball doesn't zig-zag wildly)
        x_range = max(x_vals) - min(x_vals)

        # Trajectory span (should cover significant portion of frame)
        y_span = max(y_vals) - min(y_vals)

        # Score: prefer long, smooth, downward trajectories
        score = y_span * y_ratio * len(delivery) / max(x_range * 10, 0.1)

        if score > best_score and y_ratio > 0.5 and y_span > 0.1:
            best_score = score
            best_delivery = delivery

    return best_delivery or []


def process_video(job_id):
    """Background thread: process video frames and update job progress."""
    job = jobs[job_id]
    tmp_path = job["tmp_path"]
    fps_hint = job["fps"]

    try:
        # Detect slo-mo before opening with OpenCV
        slomo_factor = detect_slomo(tmp_path)
        print(f"Slo-mo detection: factor={slomo_factor}")

        cap = cv2.VideoCapture(tmp_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or fps_hint
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Skip frames: target ~15-30 analysis frames per second of video
        # For 30fps: every 2nd frame → 15fps analysis
        # For 60fps: every 3rd frame → 20fps analysis
        # For 120fps: every 6th frame → 20fps analysis
        # For 240fps: every 12th frame → 20fps analysis
        if video_fps <= 30:
            skip = 2
        elif video_fps <= 60:
            skip = 3
        elif video_fps <= 120:
            skip = 6
        else:
            skip = 12

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

        # Use seeking for high-fps videos (much faster than read-skip)
        target_frame = 0
        while target_frame < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            if not ret:
                break

            time_s = target_frame / video_fps / slomo_factor  # Correct for slo-mo playback
            blob, scale = preprocess(frame)
            outputs = session.run(None, {input_name: blob})
            landmarks, confidence = postprocess(outputs, scale)

            frames.append({
                "frame": target_frame,
                "time": round(time_s, 4),
                "keypoints": landmarks,
                "confidence": confidence,
            })

            frames_done += 1
            job["frames_done"] = frames_done
            job["progress"] = round((frames_done / max(total_analysis, 1)) * 100)
            target_frame += skip

        cap.release()

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
    print(f"Models warmed up! Ball detection: {'ML (custom YOLOv11)' if ball_session else 'HSV fallback'}")
    # Initialize Stripe (optional — works without it)
    if init_stripe():
        print("Stripe payment system ready")
    else:
        print("Stripe not configured — payment endpoints disabled")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": session is not None}


@app.get("/")
def root():
    return {"service": "Cricket Analyze Pro - Pose API", "status": "running", "version": "2.0"}


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
