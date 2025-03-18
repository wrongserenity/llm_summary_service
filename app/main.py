from fastapi import FastAPI, HTTPException
import logging
from app.schemas import SummaryRequest, SummaryResponse
from app.model_handler import summarize_text
from app.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


@app.post("/summarize", response_model=SummaryResponse)
async def summarize(request: SummaryRequest):
    try:
        logger.info(f"Got summarization request. Text length: {len(request.text)} symbols.")
        summary = summarize_text(request.text, request.max_length)
        return {"summary": summary}
    except Exception as e:
        logger.error(f"Request processing error: {e}")
        raise HTTPException(status_code=500, detail="Summarization generation error")


@app.get("/")
async def root():
    return {"message": "Summarization service is runnins!"}
