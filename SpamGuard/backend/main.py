# backend/main.py

from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import asyncio
import json

# --- THE FIX: Use relative imports for all local modules ---
from .classifier import SpamGuardClassifier
from . import database
from . import llm_generator
from .train_nb import retrain_and_save

app = FastAPI(title="SpamGuard AI API", version="2.0.0")

@app.on_event("startup")
def startup_event():
    database.init_db()
    app.state.classifier = SpamGuardClassifier()

# The rest of the file is unchanged...
class Message(BaseModel): text: str
class Feedback(BaseModel): message: str; correct_label: str
class LLMRequest(BaseModel): provider: str; model: str; api_key: str | None = None; label_to_generate: str | None = None

@app.post("/classify")
def classify_message(message: Message):
    return app.state.classifier.classify(message.text)

@app.post("/retrain")
async def trigger_retraining():
    new_records_count = database.enrich_main_dataset()
    if new_records_count == 0:
        return {"status": "skipped", "message": "No new feedback data to train on."}
    retrain_and_save() 
    app.state.classifier.reload() 
    return {"status": "success", "message": f"Model retrained with {new_records_count} new records."}

@app.get("/")
def read_root(): return {"message": "Welcome to the SpamGuard AI API"}
@app.post("/feedback")
def receive_feedback(feedback: Feedback): database.add_feedback(message=feedback.message, label=feedback.correct_label); return {"status": "success", "message": "Feedback received."}
@app.get("/analytics")
def get_analytics(): return database.get_analytics()
@app.post("/generate_data")
async def generate_data_stream(req: LLMRequest, raw_request: Request):
    async def event_stream():
        while True:
            if await raw_request.is_disconnected(): break
            generator = None
            if req.provider == 'ollama': generator = llm_generator.generate_with_ollama(model=req.model, label_to_generate=req.label_to_generate)
            elif req.provider == 'lmstudio': generator = llm_generator.generate_with_lmstudio(model=req.model, label_to_generate=req.label_to_generate)
            elif req.provider == 'openrouter':
                if not req.api_key: yield "data: Error: OpenRouter requires an API key.\n\n"; break
                generator = llm_generator.generate_with_openrouter(model=req.model, api_key=req.api_key, label_to_generate=req.label_to_generate)
            else: yield "data: Error: Invalid provider specified.\n\n"; break
            try:
                async for status_or_data in generator:
                    if isinstance(status_or_data, dict):
                        data=status_or_data; print(f"✅ [LLM Generated] Label: {data['label']:<4} | Message: {data['message']}"); database.add_feedback(data['message'], data['label'], source='llm'); yield f"data: Generated & Saved: {json.dumps(data)}\n\n"
                    else: yield f"data: {status_or_data}\n\n"
                    await asyncio.sleep(0.1)
                yield "data: Pausing for 1.5 seconds...\n\n"; await asyncio.sleep(1.5)
            except Exception as e: error_message = f"An error occurred: {e}"; print(error_message); yield f"data: {error_message}\n\n"; break
    return StreamingResponse(event_stream(), media_type="text/event-stream")