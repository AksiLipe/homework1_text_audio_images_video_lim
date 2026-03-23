# Image object detection (YOLOv5, PyTorch Hub)

## What it does
Detects objects in images and picks the top-1 detection by confidence.

## Local model
`ultralytics/yolov5` (default `yolov5s`)

## Dataset format
`images/data/<label>/*.(jpg|jpeg|png|bmp)`
- `<label>` should match a COCO class name produced by YOLOv5 (e.g., `person`, `car`).

## Run (local script)
```powershell
.venv\Scripts\python.exe images\main.py --max-samples 50 --conf-thres 0.25
```

Results: `results/images_yolov5_*.json`

