import time
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader
import torch
import io
import os
from contextlib import asynccontextmanager

load_dotenv()

MODEL_DIR = os.getenv("MODEL_DIR", "bart-base-40000-final")
MAX_INPUT_TOKENS = 512
MAX_SUMMARY_TOKENS = 128

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("summarizer")

tokenizer = None
model = None
device = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model, device
    logger.info("Loading model from %s ...", MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    logger.info("Model loaded on device: %s", device)
 
    yield  # app runs while paused here
 
    logger.info("Shutting down, releasing model.")
    tokenizer = None
    model = None

app = FastAPI(
    title="Document Summarizer API",
    version="1.0.0",
    lifespan=lifespan,
)

class SummarizeTextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw text to summarize")

class SummarizeResponse(BaseModel):
    summary: str
    input_word_count: int
    latency_ms: float
    device: str

def summarization(input_text: str) -> tuple[str, float]:
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Please try again later.")

    start_time = time.time()
    inputs = tokenizer(
        input_text,
        return_tensors="pt", 
        max_length=MAX_INPUT_TOKENS, 
        truncation=True
    ).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            max_length=MAX_SUMMARY_TOKENS,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
        )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    latency_ms = (time.time() - start_time) * 1000
    return summary, latency_ms

# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "model_loaded": model is not None, "device": device}


@app.post("/summarize/text", tags=["Text Summarization"])
def summarize_text(request: SummarizeTextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    summary, latency = summarization(request.text)

    return SummarizeResponse(
        summary=summary,
        input_word_count=len(request.text.split()),
        latency_ms=round(latency, 1),
        device=device,
    )