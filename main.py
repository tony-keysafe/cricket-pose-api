"""Cricket Bowling Pose Analysis API using ONNX Runtime only (no PyTorch)."""
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

MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.onnx"
MODEL_PATH = "/app/yolo11n-pose.onnx"
session = None

def download_model():
    """Download ONNX model if not present."""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading model from {MODEL_URL}...")
        r = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Model downloaded: {os.path.getsize(MODEL_PATH)/1e6:.1f} MB")

def preprocess(frame, input_size=640):
    """Prepare frame for YOLO inference."""
    h, w = frame.shape[:2]
    scale = min(input_size/w, input_size/h)
    nw, nh = int(w*scale), int(h*scale)
    resized = cv2.resize(frame, (nw, nh))
    
    # Pad to square
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    padded[:nh, :nw] = resized
    
    # HWC->CHW, BGR->RGB, normalize
    blob = padded[:,:,::-1].transpose(2,0,1).astype(np.float32) / 255.0
    return np.expand_dims(blob, 0), scale, 0, 0  # batch, scale, pad_x, pad_y

def postprocess(output, scale, orig_w, orig_h, conf_thresh=0.25):
    """Extract keypoints from YOLO pose output."""
    # YOLO11 pose output shape: (1, 56, num_detections)
    # 56 = 4 (bbox) + 1 (conf) + 17*3 (keypoints x,y,conf)
    predictions = output[0]  # shape: (1, 56, N)
    if len(predictions.shape) == 3:
        predictions = predictions[0].T  # shape: (N, 56)
    
    if len(predictions) == 0:
        return None, 0
    
    # Filter by confidence
    scores = predictions[:, 4]
    mask = scores > conf_thresh
    predictions = predictions[mask]
    
    if len(predictions) == 0:
        return None, 0
    
    # Take highest confidence detection
    best_idx = predictions[:, 4].argmax()
    det = predictions[best_idx]
    
    # Extract keypoints (17 keypoints, each with x, y, conf)
    kps_raw = det[5:].reshape(17, 3)
    
    # Scale back to original image coordinates
    kps_raw[:, 0] /= scale  # x
    kps_raw[:, 1] /= scale  # y
    
    # Map COCO keypoints to MediaPipe indices
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
            blob, scale, _, _ = preprocess(frame)
            
            outputs = session.run(None, {input_name: blob})
            landmarks, confidence = postprocess(outputs, scale, width, height)

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
