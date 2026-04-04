"""Cricket Bowling Pose Analysis API - Server-side processing with ViTPose."""
import os
import io
import json
import tempfile
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
from PIL import Image

app = FastAPI(title="Cricket Analyze Pro - Pose API")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model references
pose_model = None
person_detector = None
pose_processor = None
person_processor = None

def load_models():
    """Load ViTPose and person detector on startup."""
    global pose_model, person_detector, pose_processor, person_processor
    from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation
    
    print("Loading person detector (RT-DETR)...")
    person_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
    person_detector = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
    
    print("Loading ViTPose-Base...")
    pose_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
    pose_model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple")
    
    print("Models loaded!")

@app.on_event("startup")
async def startup():
    load_models()

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": pose_model is not None}

@app.post("/analyze")
async def analyze_video(
    video: UploadFile = File(...),
    fps: int = Form(30),
    height_cm: float = Form(None),
    bowling_arm: str = Form("right"),
):
    """Process a bowling video and return pose landmarks for each frame."""
    # Save uploaded video to temp file
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
        duration = total_frames / video_fps
        
        # Determine analysis fps (skip frames for long videos)
        skip = max(1, int(video_fps / 60))  # analyze at ~60fps max
        analysis_fps = video_fps / skip
        
        results = {
            "video_info": {
                "width": width, "height": height,
                "fps": video_fps, "total_frames": total_frames,
                "duration": duration, "analysis_fps": analysis_fps,
            },
            "frames": [],
        }
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % skip != 0:
                frame_idx += 1
                continue
            
            time_s = frame_idx / video_fps
            
            # Convert to PIL Image for transformers
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Detect person
            landmarks = detect_pose(pil_image, width, height)
            
            if landmarks:
                results["frames"].append({
                    "frame": frame_idx,
                    "time": round(time_s, 4),
                    "keypoints": landmarks,
                    "confidence": np.mean([l["visibility"] for l in landmarks]),
                })
            else:
                results["frames"].append({
                    "frame": frame_idx,
                    "time": round(time_s, 4),
                    "keypoints": None,
                    "confidence": 0,
                })
            
            frame_idx += 1
        
        cap.release()
        
        return JSONResponse(content=results)
    
    finally:
        os.unlink(tmp_path)

def detect_pose(pil_image, img_w, img_h):
    """Detect person and estimate pose for a single frame."""
    if pose_model is None:
        return None
    
    # 1. Detect person
    inputs = person_processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        outputs = person_detector(**inputs)
    
    det_results = person_processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([(img_h, img_w)]),
        threshold=0.3
    )[0]
    
    # Filter for person class (class 0 in COCO)
    person_mask = det_results["labels"] == 0
    if not person_mask.any():
        return None
    
    person_boxes = det_results["boxes"][person_mask]
    person_scores = det_results["scores"][person_mask]
    
    # Take highest confidence person
    best_idx = person_scores.argmax()
    bbox = person_boxes[best_idx].unsqueeze(0)
    
    # 2. Estimate pose
    pose_inputs = pose_processor(images=pil_image, boxes=[bbox], return_tensors="pt")
    with torch.no_grad():
        pose_outputs = pose_model(**pose_inputs)
    
    # Post-process
    pose_results = pose_processor.post_process_pose_estimation(
        pose_outputs,
        boxes=[bbox],
        threshold=0.3
    )
    
    if not pose_results or len(pose_results[0]) == 0:
        return None
    
    # Convert to our format (matching MediaPipe landmark indices)
    # COCO keypoints: nose, left_eye, right_eye, left_ear, right_ear,
    # left_shoulder, right_shoulder, left_elbow, right_elbow,
    # left_wrist, right_wrist, left_hip, right_hip,
    # left_knee, right_knee, left_ankle, right_ankle
    
    # Map COCO indices to MediaPipe indices
    COCO_TO_MP = {
        0: 0,    # nose
        5: 11,   # left_shoulder
        6: 12,   # right_shoulder
        7: 13,   # left_elbow
        8: 14,   # right_elbow
        9: 15,   # left_wrist
        10: 16,  # right_wrist
        11: 23,  # left_hip
        12: 24,  # right_hip
        13: 25,  # left_knee
        14: 26,  # right_knee
        15: 27,  # left_ankle
        16: 28,  # right_ankle
    }
    
    kps = pose_results[0][0]  # First person, first result
    keypoints_list = kps.get("keypoints", [])
    scores_list = kps.get("scores", [])
    
    # Build MediaPipe-compatible output (33 landmarks, null for unmapped)
    mp_landmarks = [None] * 33
    for coco_idx, mp_idx in COCO_TO_MP.items():
        if coco_idx < len(keypoints_list):
            kp = keypoints_list[coco_idx]
            sc = scores_list[coco_idx].item() if coco_idx < len(scores_list) else 0.5
            mp_landmarks[mp_idx] = {
                "x": round(kp[0].item(), 1),
                "y": round(kp[1].item(), 1),
                "z": 0,
                "visibility": round(sc, 3),
            }
    
    return mp_landmarks

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
