import time
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader
import torch
import io

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("summarizer")
 
app = FastAPI(
    title="Document Summarizer API",
    version="1.0.0"
)

tokenizer = None
model = None
device = None

@app.on_event("startup")
def load_model():
    global tokenizer, model, device
    logging.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    logger.info("Model loaded on device: %s", device)

# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "model_loaded": model is not None, "device": device}