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
BALL_MODEL_URLS = [
    "https://huggingface.co/Xenova/yolov8s/resolve/main/onnx/model.onnx",
    "https://huggingface.co/qualcomm/YOLOv8-Detection-Nano/resolve/main/YOLOv8-Detection-Nano.onnx",
]
BALL_MODEL_PATH = "/app/yolov8-detect.onnx"
INPUT_SIZE = 640  # Model has fixed 640x640 input dimensions
session = None
ball_session = None

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
    # Pose model (required)
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1000000:
        print(f"Pose model already exists: {os.path.getsize(MODEL_PATH)/1e6:.1f} MB")
    else:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        print(f"Downloading pose model...")
        r = requests.get(MODEL_URL, allow_redirects=True, stream=True)
        r.raise_for_status()
        with open(MODEL_PATH, 'wb') as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
        print(f"Pose model: {os.path.getsize(MODEL_PATH)/1e6:.1f} MB")

    # Ball detection model (optional — try multiple sources)
    if os.path.exists(BALL_MODEL_PATH) and os.path.getsize(BALL_MODEL_PATH) > 1000000:
        print(f"Ball model already exists: {os.path.getsize(BALL_MODEL_PATH)/1e6:.1f} MB")
    else:
        for url in BALL_MODEL_URLS:
            try:
                print(f"Trying ball model: {url}")
                r = requests.get(url, allow_redirects=True, stream=True, timeout=30)
                r.raise_for_status()
                with open(BALL_MODEL_PATH, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        f.write(chunk)
                size = os.path.getsize(BALL_MODEL_PATH)
                if size > 1000000:
                    print(f"Ball model downloaded: {size/1e6:.1f} MB from {url}")
                    break
                else:
                    os.remove(BALL_MODEL_PATH)
            except Exception as e:
                print(f"Ball model download failed from {url}: {e}")
                if os.path.exists(BALL_MODEL_PATH):
                    os.remove(BALL_MODEL_PATH)
        if not os.path.exists(BALL_MODEL_PATH):
            print("WARNING: Ball detection model not available — /detect-ball will use frame differencing fallback")


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
    """Background thread: detect ball using ML model or frame-diff fallback with bowler masking."""
    job = jobs[job_id]
    tmp_path = job["tmp_path"]
    use_ml = ball_session is not None

    try:
        slomo_factor = detect_slomo(tmp_path)
        cap = cv2.VideoCapture(tmp_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        skip = max(1, int(video_fps / 30))  # Target ~30 analysis fps
        total_analysis = total_frames // skip
        job["video_info"] = {
            "width": width, "height": height,
            "fps": round(video_fps, 2),
            "total_frames": total_frames,
            "slomo_factor": slomo_factor,
            "real_duration": round(total_frames / video_fps / slomo_factor, 3),
            "method": "yolov8" if use_ml else "frame_diff_masked",
        }
        job["total_frames"] = total_analysis
        job["status"] = "processing"

        ball_positions = []
        frames_done = 0
        prev_gray = None

        target_frame = 0
        while target_frame < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            if not ret:
                break

            time_s = target_frame / video_fps / slomo_factor

            if use_ml:
                # ML ball detection
                input_name = ball_session.get_inputs()[0].name
                blob, scale = preprocess(frame)
                outputs = ball_session.run(None, {input_name: blob})
                detections = postprocess_ball(outputs, scale, width, height)
                if detections:
                    best = detections[0]
                    ball_positions.append({
                        "frame": target_frame, "time": round(time_s, 4),
                        "x": best["nx"], "y": best["ny"],
                        "px": best["x"], "py": best["y"],
                        "conf": best["confidence"],
                    })
            else:
                # Frame-diff fallback: use pose model to find bowler, mask them, detect ball
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)

                if prev_gray is not None:
                    # Get bowler position from pose model to mask them
                    input_name = session.get_inputs()[0].name
                    blob, scale = preprocess(frame)
                    outputs = session.run(None, {input_name: blob})
                    landmarks, confidence = postprocess(outputs, scale)

                    # Create mask excluding the bowler (expand bounding box around detected person)
                    mask = np.ones_like(gray, dtype=np.uint8) * 255
                    if landmarks and confidence > 0.3:
                        xs = [lm["x"] for lm in landmarks if lm and lm.get("visibility", 0) > 0.2]
                        ys = [lm["y"] for lm in landmarks if lm and lm.get("visibility", 0) > 0.2]
                        if xs and ys:
                            pad = int(min(width, height) * 0.08)
                            x1 = max(0, int(min(xs)) - pad)
                            y1 = max(0, int(min(ys)) - pad)
                            x2 = min(width, int(max(xs)) + pad)
                            y2 = min(height, int(max(ys)) + pad)
                            mask[y1:y2, x1:x2] = 0  # Mask out bowler area

                    # Frame difference with bowler masked
                    diff = cv2.absdiff(prev_gray, gray)
                    diff = cv2.bitwise_and(diff, mask)
                    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

                    # Find contours — ball should be small and compact
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    best_ball = None
                    best_score = 0
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area < 5 or area > width * height * 0.01:  # Ball is small
                            continue
                        x, y, bw, bh = cv2.boundingRect(cnt)
                        aspect = max(bw, bh) / max(min(bw, bh), 1)
                        if aspect > 3:  # Ball is roughly circular
                            continue
                        cx = (x + bw / 2) / width
                        cy = (y + bh / 2) / height
                        # Prefer compact, central objects
                        score = area * (1 if 0.2 < cx < 0.8 else 0.3)
                        if score > best_score:
                            best_score = score
                            best_ball = {"x": round(cx, 4), "y": round(cy, 4), "area": area}

                    if best_ball:
                        ball_positions.append({
                            "frame": target_frame, "time": round(time_s, 4),
                            "x": best_ball["x"], "y": best_ball["y"],
                            "conf": 0.5,
                        })

                prev_gray = gray

            frames_done += 1
            job["frames_done"] = frames_done
            job["progress"] = round((frames_done / max(total_analysis, 1)) * 100)
            target_frame += skip

        cap.release()
        job["ball_positions"] = ball_positions
        job["video_info"]["frames_analyzed"] = frames_done
        job["video_info"]["ball_detections"] = len(ball_positions)
        job["status"] = "complete"
        job["progress"] = 100
        print(f"Ball job {job_id}: {len(ball_positions)} detections in {frames_done} frames ({job['video_info']['method']})")

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
    print("Loading ONNX models...")
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    print(f"Pose model loaded! Input: {session.get_inputs()[0].shape}")

    if os.path.exists(BALL_MODEL_PATH) and os.path.getsize(BALL_MODEL_PATH) > 1000000:
        try:
            ball_session = ort.InferenceSession(BALL_MODEL_PATH, providers=['CPUExecutionProvider'])
            print(f"Ball detection model loaded! Input: {ball_session.get_inputs()[0].shape}")
        except Exception as e:
            print(f"Ball model failed to load: {e}")
            ball_session = None
    else:
        print("Ball detection model not available — will use frame-diff fallback")

    # Warm up pose model
    dummy = np.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
    session.run(None, {session.get_inputs()[0].name: dummy})
    if ball_session:
        ball_session.run(None, {ball_session.get_inputs()[0].name: dummy})
    print("Models warmed up!")
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
