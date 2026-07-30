# Important commands

Project root (CMD/PowerShell): `d:\Data Mining project\Livestock-Cattle-Weight-Estimation`

Project root (Git Bash): `/d/Data Mining project/Livestock-Cattle-Weight-Estimation`

## URLs

| Service | URL |
|---------|-----|
| Frontend (Vite) | http://127.0.0.1:5173 |
| Backend API | http://127.0.0.1:5001 |

## First-time setup

**CMD / PowerShell**

```bat
cd "d:\Data Mining project\Livestock-Cattle-Weight-Estimation"
.\.venv\Scripts\activate

cd frontend_dashboard\frontend
npm install
```

**Git Bash**

```bash
cd "/d/Data Mining project/Livestock-Cattle-Weight-Estimation"
source .venv/Scripts/activate

cd frontend_dashboard/frontend
npm install
```

Python deps use the project-root `.venv` (see `requirements.txt` at repo root if you need to reinstall).

## One-command start

Starts backend + frontend together:

**CMD / PowerShell**

```bat
cd "d:\Data Mining project\Livestock-Cattle-Weight-Estimation"
.\.venv\Scripts\activate
python frontend_dashboard\run_dashboard.py
```

**Git Bash**

```bash
cd "/d/Data Mining project/Livestock-Cattle-Weight-Estimation"
source .venv/Scripts/activate
python frontend_dashboard/run_dashboard.py
```

Or double-click:

```bat
frontend_dashboard\start_dashboard.bat
```

## Manual start (two terminals)

### Terminal 1 — backend

**CMD / PowerShell**

```bat
cd "d:\Data Mining project\Livestock-Cattle-Weight-Estimation"
.\.venv\Scripts\activate
python frontend_dashboard\backend\app.py
```

**Git Bash**

```bash
cd "/d/Data Mining project/Livestock-Cattle-Weight-Estimation"
source .venv/Scripts/activate
python frontend_dashboard/backend/app.py
```

### Terminal 2 — frontend

**CMD / PowerShell**

```bat
cd "d:\Data Mining project\Livestock-Cattle-Weight-Estimation\frontend_dashboard\frontend"
npm start
```

**Git Bash**

```bash
cd "/d/Data Mining project/Livestock-Cattle-Weight-Estimation/frontend_dashboard/frontend"
npm start
```

(`npm run dev` is the same as `npm start`.)

## Frontend build

**CMD / PowerShell**

```bat
cd "d:\Data Mining project\Livestock-Cattle-Weight-Estimation\frontend_dashboard\frontend"
npm run build
```

**Git Bash**

```bash
cd "/d/Data Mining project/Livestock-Cattle-Weight-Estimation/frontend_dashboard/frontend"
npm run build
```

## Restart backend after Python changes

Flask runs with `debug=False` and does **not** auto-reload. After editing backend / morpho / segmentation code:

**PowerShell — stop port 5001, then start again**

```powershell
$conn = Get-NetTCPConnection -LocalPort 5001 -ErrorAction SilentlyContinue | Select-Object -First 1
if ($conn) { Stop-Process -Id $conn.OwningProcess -Force }
cd "d:\Data Mining project\Livestock-Cattle-Weight-Estimation"
.\.venv\Scripts\python.exe frontend_dashboard\backend\app.py
```

**Git Bash — stop port 5001, then start again**

```bash
# Find and kill process on port 5001 (if any)
pid=$(netstat -ano | grep ':5001' | grep LISTENING | awk '{print $5}' | head -n 1)
if [ -n "$pid" ]; then taskkill //F //PID "$pid"; fi

cd "/d/Data Mining project/Livestock-Cattle-Weight-Estimation"
source .venv/Scripts/activate
python frontend_dashboard/backend/app.py
```

Or Ctrl+C in the backend terminal, then re-run the backend start command above.

## Results

Each run is stored under:

```text
frontend_dashboard/results/<run_id>/
```

## Git hooks (no Cursor/AI trailers on GitHub)

Once per clone, from the repo root:

```bash
git config core.hooksPath githooks
```

See [`githooks/README.md`](githooks/README.md). `commit-msg` strips Cursor/AI footers; `pre-push` blocks pushes that still contain them.

## Vercel (frontend only)

The Flask/YOLO API is **not** deployed on Vercel (pose model is too large for serverless). Deploy the Vite UI only:

1. Import [https://github.com/SASagorAhmed/Livestock-Cattle-Weight-Estimation](https://github.com/SASagorAhmed/Livestock-Cattle-Weight-Estimation) in Vercel.
2. Root directory: repository root (uses [`vercel.json`](vercel.json)).
3. Build uses `frontend_dashboard/frontend`; output is `frontend_dashboard/frontend/dist`.
4. Set env **`VITE_API_BASE`** to your public API URL (no trailing slash), e.g. `https://your-api.example.com`. Local default is `http://127.0.0.1:5001` (see `frontend_dashboard/frontend/.env.example`).

Large model weights use **Git LFS** (`*.onnx`, `*.onnx.data`, `*.pt`, `*.h5`, `*.pkl`). After clone: `git lfs pull`.

Pose inference needs the vendored dependency (not in this repo; nested clone stays local):

```bash
git clone https://github.com/JunkyByte/easy_ViTPose.git cow_pose_detection/easy_ViTPose
```

## Backend (local Flask API)

```bash
cd frontend_dashboard/backend
# from repo root: activate .venv first
python app.py
```