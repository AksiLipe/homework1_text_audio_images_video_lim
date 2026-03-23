# Text sentiment model (FastAPI + Transformers)

## What it does
Predicts sentiment for a text as `positive` / `negative` / `neutral`.

## Local model
`tabularisai/multilingual-sentiment-analysis`

## Dataset format
`text/data/<label>/*.txt`
- `<label>` should match the model outputs (case-insensitive): `positive`, `negative`, `neutral`.

## Run (local script)
```powershell
.venv\Scripts\python.exe text\main.py --max-samples 50
```

## Run (API)
```powershell
.venv\Scripts\python.exe text\app.py
```
- Swagger: `http://localhost:8000/docs`
- Endpoint: `POST /predict` with JSON `{"text":"..."}`.

