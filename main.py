"""Cricket Bowling Pose Analysis API using ONNX Runtime only."""
import os
import tempfile
import numpy as np
import cv2
import onnxruntime as ort
import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="Cricket Analyze Pro - Pose API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_URL = "https://huggingface.co/Xenova/yolov8-pose-onnx/resolve/main/yolov8n-pose.onnx"
MODEL_PATH = "/app/yolov8n-pose.onnx"
session = None

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

def preprocess(frame, input_size=640):
    h, w = frame.shape[:2]
    scale = min(input_size/w, input_size/h)
    nw, nh = int(w*scale), int(h*scale)
    resized = cv2.resize(frame, (nw, nh))
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    padded[:nh, :nw] = resized
    blob = padded[:,:,::-1].transpose(2,0,1).astype(np.float32) / 255.0
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

@app.on_event("startup")
async def startup():
    global session
    download_model()
    print("Loading ONNX model...")
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    print(f"Model loaded! Input: {session.get_inputs()[0].shape}")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": session is not None}

@app.get("/")
def root():
    return {"service": "Cricket Analyze Pro - Pose API", "status": "running"}

@app.post("/analyze")
async def analyze_video(
    video: UploadFile = File(...),
    fps: int = Form(30),
    height_cm: float = Form(0),
    bowling_arm: str = Form("right"),
):
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
