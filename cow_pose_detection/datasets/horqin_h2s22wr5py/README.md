# Horqin cattle biometric dataset (Mendeley)

**DOI:** 10.17632/h2s22wr5py.2  
**URL:** https://data.mendeley.com/datasets/h2s22wr5py/2  
**Paper:** Image dataset for cattle biometric detection and analysis

## Expected contents (after you download + extract)

| Item | Present? |
|------|----------|
| Side-view cattle images (PNG) | inspect |
| Back-view cattle images (PNG) | inspect |
| LabelMe JSON annotations | inspect |
| Excel/XLSX measurement table | inspect |

## Setup (existing project only)

1. Download **Download All** from Mendeley (browser).
2. Extract into:
   `cow_pose_detection/datasets/horqin_h2s22wr5py/raw/`
3. Inspect:
   ```bash
   cd cow_pose_detection/datasets
   python inspect_dataset.py
   ```
4. If labels ≠ the 4 target keypoints (likely), prepare LabelMe workspace:
   ```bash
   python prepare_labelme_workspace.py
   ```
5. Relabel side views in LabelMe (see `horqin_h2s22wr5py/labelme_workspace/HOW_TO_RELABEL.md`).
6. Convert + train:
   ```bash
   python labelme_to_yolo_pose.py
   python train_yolo_pose.py
   ```
7. Weights are copied to `cow_pose_detection/models/four_point_pose.pt` for the app.

## Target keypoints

0. `A_start_lower_chest`
1. `A_end_withers`
2. `B_start_tail_head`
3. `B_end_forward_shoulder_lower`

These feed the **Smartphone Diagonal Formula** path only — never `h5model.h5`.
