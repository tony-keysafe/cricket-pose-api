# Cricket Analyze Pro - Pose API Backend

Server-side pose estimation using ViTPose for more accurate bowling analysis.

## Deploy to Railway

1. Create account at [railway.app](https://railway.app)
2. Create a new GitHub repo called `cricket-pose-api`
3. Push this code to that repo
4. In Railway, click "New Project" → "Deploy from GitHub Repo" → select `cricket-pose-api`
5. Railway will build and deploy automatically
6. Copy the deployment URL (e.g. `https://cricket-pose-api-production.up.railway.app`)
7. Set this URL in your frontend app's settings

## API Endpoints

- `GET /health` - Check if server is running and models are loaded
- `POST /analyze` - Upload video for analysis
  - Form fields: `video` (file), `fps` (int), `height_cm` (float), `bowling_arm` (string)
  - Returns: JSON with frame-by-frame keypoint data

## Local Development

```bash
pip install -r requirements.txt
python main.py
```

Server runs on port 8000 by default.

## Notes

- First request after deployment takes ~60s as models download from HuggingFace
- Subsequent requests use cached models
- Uses RT-DETR for person detection + ViTPose-Base for pose estimation
- CPU-only (no GPU required) — ~50ms per frame
- Railway free tier: 500 hours/month, 8GB RAM (sufficient for this)
