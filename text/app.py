from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import pipeline

app = FastAPI(title="Text Sentiment API")


class TextRequest(BaseModel):
    text: str


device = 0 if torch.cuda.is_available() else -1
pipe = pipeline(
    "text-classification",
    model="tabularisai/multilingual-sentiment-analysis",
    device=device,
)


@app.get("/")
def root():
    return {"message": "Text Sentiment API"}


@app.post("/predict")
def predict(req: TextRequest):
    pred = pipe(req.text)[0]
    return {"text": req.text, "prediction": pred}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

