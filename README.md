## dz1 — приложения с готовыми ML-моделями (4 типа контента + LLM)

### Важно про данные
Скрипты ниже рассчитаны на запуск:
1) с твоими локальными данными (если они лежат в папках `text/data`, `audio/data`, `images/data`, `video/data`)
2) как fallback — на публичных датасетах/примере (если локально пусто).

Сейчас в `*/data` у тебя пусто, поэтому для проверки метрик сначала положи данные в папки или запускай fallback.

### Структура данных (минимальные ожидания)
1) `text/data/<label>/*.txt` — файл содержит текст. `<label>` должен совпадать с классами модели `tabularisai/multilingual-sentiment-analysis` (`positive`/`negative`/`neutral`, регистр не важен).
2) `audio/data/*.wav` — аудио для классификации событий (YAMNet). Если рядом есть `имя.wav.txt` с ground-truth `display_name` (как в YAMNet), посчитается accuracy.
3) `images/data/<label>/*.(jpg|jpeg|png|bmp)` — картинка. `<label>` должен совпадать с одним из классов, которые предсказывает YOLOv5 (COCO, например `person`, `car`).
4) `video/data/<label>/*.mp4` — видео; `<label>` должен совпадать с одним из классов жестов модели `prithivMLmods/Hand-Gesture-19`.

### Установка зависимостей
В корне проекта:
```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Тонкость: для аудио используется TensorFlow Hub (YAMNet), поэтому `tensorflow` и `tensorflow-hub` должны успешно установиться (может занять время при первом `pip install`).

Если у тебя будут проблемы с лимитами/скоростью скачивания моделей, задай `HF_TOKEN`:
```powershell
$env:HF_TOKEN="твой_токен"
```

### Запуск
1) Текст (тональность, локально):
```powershell
.venv\Scripts\python.exe text\main.py --max-samples 50
```

2) Аудио (YAMNet):
```powershell
.venv\Scripts\python.exe audio\main.py --max-samples 10
```

3) Изображения (YOLOv5):
```powershell
.venv\Scripts\python.exe images\main.py --max-samples 50 --conf-thres 0.25
```

4) Видео (жесты, классификация кадров SigLIP):
```powershell
.venv\Scripts\python.exe video\main.py --max-samples 10 --num-frames 16 --download-sample
```

5) LLM (локальный генератор):
```powershell
.venv\Scripts\python.exe llm\main.py --prompt "Кратко объясни разницу между ML и DL."
```

6) Текст API (FastAPI):
```powershell
.venv\Scripts\python.exe text\app.py
```

Все результаты пишутся в папку `results/`.

Для оформления отчёта смотри `REPORT.md` (там структура и что нужно заполнить своими результатами).
