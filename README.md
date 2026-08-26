# Document Summarizer

A fine-tuned BART-base model for abstractive text summarization, served via a FastAPI backend.

Fine-tuned on 40,000 examples from the CNN/DailyMail dataset, achieving:

| Metric | Score |
|---|---|
| ROUGE-1 | 0.358 |
| ROUGE-2 | 0.147 |
| ROUGE-L | 0.247 |
| ROUGE-Lsum | 0.331 |

(Evaluated on a 2,000-example held-out validation set never seen during training.)

## What this project does

- Fine-tunes `facebook/bart-base` on CNN/DailyMail news articles to generate abstractive summaries
- Serves the model via a FastAPI backend with two endpoints: raw text summarization and PDF upload summarization
- Exposes interactive Swagger docs (`/docs`) for live testing without needing a frontend

## Project structure

```
document-summarizer/
├── train_summarizer.ipynb    # Colab notebook: dataset prep, fine-tuning, evaluation
├── main.py                   # FastAPI app serving the trained model
├── requirements.txt
├── .env.example               # Copy to .env and configure locally
└── bart-base-40000-final/     # Trained model weights (not included in repo — see below)
```

## Model weights are not included in this repo

The trained model (`bart-base-40000-final/`) is excluded via `.gitignore` because model weights are too large for a git repository. To get a working model, either:

**Option A — Train it yourself (recommended, ~2 hours on a free Colab T4 GPU):**
1. Open `train_summarizer.ipynb` in [Google Colab](https://colab.research.google.com)
2. Runtime → Change runtime type → GPU
3. Run all cells top to bottom
4. Download the resulting `bart-base-40000-final.zip` and unzip it into the project root

**Option B — Use a smaller/faster local run for testing:**
Edit `TRAIN_SIZE` in the notebook's config cell down to something like `2000` for a much faster (but lower-quality) local sanity check.

## Setup

```bash
git clone <your-repo-url>
cd document-summarizer

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Running the API

```bash
uvicorn app:app --reload
```

Then open **http://localhost:8000/docs** for interactive Swagger UI, or hit the endpoints directly:

```bash
curl -X POST http://localhost:8000/summarize/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Your long article text here..."}'
```

```bash
curl -X POST http://localhost:8000/summarize/pdf \
  -F "file=@/path/to/document.pdf"
```

## Design notes / known limitations

- Input text is truncated to the first 512 tokens (~350–400 words). Longer documents are summarized based only on their opening section — this is a deliberate speed/simplicity tradeoff, not an oversight.
- Trained on `bart-base` (not `bart-large`) and a 40K-example subset (not the full ~287K CNN/DailyMail dataset) to keep training time practical on free-tier Colab GPUs. This is why ROUGE scores are lower than published `bart-large` benchmarks (~0.44 ROUGE-1) — a larger model and full dataset would likely close much of that gap.
- Inference runs on CPU/MPS locally (no GPU required to *use* the model, only to train it).

## Possible next steps

- Fine-tune `bart-large` on the same data for a stronger quality/latency tradeoff comparison
- Add batch summarization endpoint
- Add automated evaluation script (`evaluate.py`) decoupled from the training notebook