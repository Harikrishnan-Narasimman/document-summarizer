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
import re
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

def clean_summary(text: str) -> str:
    """Fix common BART detokenization artifacts (space before punctuation, merged words)."""
    text = re.sub(r"\s+([.,!?])", r"\1", text)  # "complete ." -> "complete."
    text = re.sub(r"\n+", " ", text)             # collapse stray newlines from generation
    text = re.sub(r"\s{2,}", " ", text)          # collapse double spaces
    return text.strip()

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

    raw_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    summary = clean_summary(raw_summary)
    latency_ms = (time.time() - start_time) * 1000
    return summary, latency_ms

def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()

# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "model_loaded": model is not None, "device": device}


@app.post("/summarize/text", tags=["Summarization"])
def summarize_text(request: SummarizeTextRequest):
    """
    Summarize raw text input.
 
    Note: input is truncated to the first 512 tokens (~350-400 words);
    longer documents will only be summarized based on their opening section.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    summary, latency = summarization(request.text)

    return SummarizeResponse(
        summary=summary,
        input_word_count=len(request.text.split()),
        latency_ms=round(latency, 1),
        device=device,
    )

@app.post("/summarize/pdf", response_model=SummarizeResponse, tags=["Summarization"])
async def summarize_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF and get back a summary of its extracted text.
 
    Note: input is truncated to the first 512 tokens (~350-400 words);
    longer documents will only be summarized based on their opening section.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
 
    file_bytes = await file.read()
    extracted_text = extract_text_from_pdf_bytes(file_bytes)
 
    if not extracted_text:
        raise HTTPException(
            status_code=422,
            detail="Could not extract text from this PDF. It may be a scanned image without OCR.",
        )
 
    summary, latency = summarization(extracted_text)
 
    return SummarizeResponse(
        summary=summary,
        input_word_count=len(extracted_text.split()),
        latency_ms=round(latency, 1),
        device=device,
    )