# Audio event classification (YAMNet, TensorFlow Hub)

## What it does
Classifies audio events using YAMNet and outputs top-k classes + confidence.

## Local model
`https://tfhub.dev/google/yamnet/1`

## Dataset format
`audio/data/*.wav`

Optional ground truth:
- sidecar `audio/data/<...>/name.txt`
- file contains YAMNet `display_name` (ground truth top-1 label).

## Run (local script)
```powershell
.venv\Scripts\python.exe audio\main.py --max-samples 10
```

Results: `results/audio_yamnet_*.json`

