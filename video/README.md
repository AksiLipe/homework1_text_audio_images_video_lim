# Video gesture classification (SigLIP)

## What it does
Samples frames from each video, classifies each frame as a hand gesture, and
uses majority vote to get the final prediction for the video.

## Local model
`prithivMLmods/Hand-Gesture-19` (SigLIP)

## Dataset format
`video/data/<label>/*.mp4` (or mkv/avi/mov/webm)
- `<label>` should match one of the model gesture labels (`model.config.id2label`).

## Run (local script)
```powershell
.venv\Scripts\python.exe video\main.py --max-samples 10 --num-frames 16 --download-sample
```

Results: `results/video_gesture_*.json`

