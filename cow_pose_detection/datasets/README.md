# Datasets for Cow Weight Detection

## Horqin cattle biometric set (Mendeley)

- DOI: `10.17632/h2s22wr5py.2`
- URL: https://data.mendeley.com/datasets/h2s22wr5py/2

**Status on this machine:** not downloaded yet (folder is empty except README).

### Download + train 4-keypoint YOLO-pose

1. Browser-download from Mendeley → extract into  
   `cow_pose_detection/datasets/horqin_h2s22wr5py/raw/`
2. `python inspect_dataset.py`
3. `python prepare_labelme_workspace.py`  (relabel if labels ≠ 4 targets)
4. LabelMe → place the 4 points on **side views**
5. `python labelme_to_yolo_pose.py`
6. `python train_yolo_pose.py` → copies weights to `cow_pose_detection/models/four_point_pose.pt`

### App usage (existing frontend only)

Upload → choose **Smartphone Diagonal Formula (Experimental)**  
→ place reference scale + 4 points (auto-suggest when model exists)  
→ formula weight is separate from `h5model.h5`.
