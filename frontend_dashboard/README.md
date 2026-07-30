# Local Cattle Weight Estimation Dashboard

Fully offline React + Flask dashboard that reuses existing project modules.

## URLs

- Frontend (Vite): http://127.0.0.1:5173
- Backend API: http://127.0.0.1:5001

## One-command start

```bat
cd "d:\Data Mining project\Livestock-Cattle-Weight-Estimation"
.\.venv\Scripts\activate
python frontend_dashboard\run_dashboard.py
```

Or double-click `frontend_dashboard\start_dashboard.bat`.

## Manual start

Terminal 1 — backend:

```bat
cd "d:\Data Mining project\Livestock-Cattle-Weight-Estimation"
.\.venv\Scripts\activate
python frontend_dashboard\backend\app.py
```

Terminal 2 — frontend:

```bat
cd "d:\Data Mining project\Livestock-Cattle-Weight-Estimation\frontend_dashboard\frontend"
npm install
npm run dev
```

## First-time frontend install

```bat
cd frontend_dashboard\frontend
npm install
```

Dependencies are installed locally via npm (no CDN).

## Pipeline steps

1. Upload image (JPG/PNG, optional webcam, segmentation toggle, prediction mode)
2. Cow detection (YOLO boxes, pick primary / alternate cow)
3. Pose keypoints (ViTPose AP-10K, 17 points)
4. Body measurements (measurements.py)
5. Pixel calculation formulas with real substituted values
6. Segmentation (optional YOLO-seg)
7. Reference scale px→cm
8. Normalized ratios
9. Weight prediction (heuristic baseline; measurement model if pkl exists)
10. Final report + downloads

## Outputs

Each run is stored under:

`frontend_dashboard/results/<run_id>/`

Models under the project root and `cow_pose_detection/models/` are never modified.
