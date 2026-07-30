# Cow Body Keypoint Detection

Offline cow body keypoint detection using **ViTPose** (AP-10K pretrained) + **YOLOv8** for bounding box detection.

## Detected Keypoints (17 AP-10K)

| Index | Keypoint           |
|-------|--------------------|
| 0     | left_eye           |
| 1     | right_eye          |
| 2     | nose               |
| 3     | neck               |
| 4     | tail_root          |
| 5     | left_shoulder      |
| 6     | left_elbow         |
| 7     | left_front_hoof    |
| 8     | right_shoulder     |
| 9     | right_elbow        |
| 10    | right_front_hoof   |
| 11    | left_hip           |
| 12    | left_knee          |
| 13    | left_back_hoof     |
| 14    | right_hip          |
| 15    | right_knee         |
| 16    | right_back_hoof    |

## Setup

```bash
cd cow_pose_detection
# Activate the project venv
..\.venv\Scripts\activate  # Windows
# or: source ../.venv/bin/activate  # Linux/Mac
```

Models are already in `models/`:
- `vitpose-b-ap10k.onnx` — ViTPose-Base trained on AP-10K (360 MB)
- `yolov8s.onnx` — YOLOv8s object detector (45 MB)

## Usage

### Single image
```bash
python detect_cow_pose.py --input path/to/cow.jpg
```

### Video file
```bash
python detect_cow_pose.py --input path/to/video.mp4 --save-video
```

### Webcam
```bash
python detect_cow_pose.py --webcam 0
# Press 's' to save snapshot, 'q' to quit
```

### Options
```
--output DIR        Output directory (default: ./output)
--conf FLOAT        Keypoint confidence threshold (default: 0.3)
--yolo-size INT     YOLO input size (default: 640)
--show              Display results in a window
--no-labels         Don't draw keypoint names
--no-save-img       Don't save annotated image
--no-save-json      Don't save JSON
--save-video        Save annotated video
```

## Output

- **Annotated images**: `output/<name>_pose.png` with skeleton, keypoints, and bounding boxes
- **JSON files**: `output/<name>_keypoints.json` with per-keypoint (x, y, confidence)

---

## Body measurements (from pose JSON)

Reads existing `*_keypoints.json`, selects the **primary cow** (largest bbox), and computes pixel morphometrics + size-normalized ratios. Pose/weight models are not modified.

### Single image
```bash
python measure_cow.py --pose output/img_9_keypoints.json --image ../uploads/img_9.jpg
```

### With optional local YOLO segmentation
```bash
python measure_cow.py --pose output/img_9_keypoints.json --image ../uploads/img_9.jpg --segment
```

### With reference marker (pixels → cm)
```bash
# Known marker is 20 cm and measures 100 px in the image
python measure_cow.py --pose output/img_9_keypoints.json --image ../uploads/img_9.jpg --ref-px 100 --ref-cm 20

# Or pass scale directly
python measure_cow.py --pose output/img_9_keypoints.json --image ../uploads/img_9.jpg --scale-cm-per-px 0.2
```

### Batch folder → CSV
```bash
# Uses existing keypoints JSON in output/
python batch_measure.py --images ../uploads --pose-dir output --output output

# Run pose first if JSON is missing, then measure (+ optional seg)
python batch_measure.py --images ../uploads --run-pose --segment --output output
```

### Measurements computed

| Pixel metric | Definition |
|--------------|------------|
| shoulder_width | left_shoulder ↔ right_shoulder |
| hip_width | left_hip ↔ right_hip |
| body_length | shoulder_center → hip_center |
| body_height | back top → mean hoof ground line |
| left/right front-leg length | shoulder → elbow → front hoof |
| left/right back-leg length | hip → knee → back hoof |
| chest_depth_proxy | shoulder_center → elbow_center |
| torso_diagonal | mean of L_shoulder→R_hip and R_shoulder→L_hip |

**Ratios:** body_length/body_height, shoulder_width/body_length, hip_width/body_length, chest_depth/body_height

**Optional segmentation:** body_pixel_area, torso_pixel_area, body_perimeter_px, belly_boundary_points

### Measurement outputs

- `output/<name>_measurements.json`
- `output/<name>_measurements.png` (measurement lines + labels)
- `output/batch_measurements.csv` (from `batch_measure.py`)

## Runs fully offline

All inference uses local ONNX/PT models. No API calls, no cloud services.
