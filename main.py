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
INPUT_SIZE = 640  # Model has fixed 640x640 input dimensions
session = None

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
        print(f"Model already exists: {os.path.getsize(MODEL_PATH)/1e6:.1f} MB")
        return
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    print(f"Downloading model from {MODEL_URL}...")
    r = requests.get(MODEL_URL, allow_redirects=True, stream=True)
    r.raise_for_status()
    total = 0
    with open(MODEL_PATH, 'wb') as f:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
            total += len(chunk)
    size = os.path.getsize(MODEL_PATH)
    print(f"Model downloaded: {size/1e6:.1f} MB ({total} bytes written)")
    if size < 1000000:
        raise RuntimeError(f"Model file too small ({size} bytes), download likely failed")


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


def process_video(job_id):
    """Background thread: process video frames and update job progress."""
    job = jobs[job_id]
    tmp_path = job["tmp_path"]
    fps_hint = job["fps"]

    try:
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
            "skip_factor": skip,
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

            time_s = target_frame / video_fps
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
    global session
    download_model()
    print("Loading ONNX model...")
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    print(f"Model loaded! Input: {session.get_inputs()[0].shape}")
    # Warm up with a dummy inference
    dummy = np.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
    session.run(None, {session.get_inputs()[0].name: dummy})
    print("Model warmed up!")
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
