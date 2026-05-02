"""
main.py
-------
ENA OBE ASSISTANT — FastAPI Entry Point

Run:
    uvicorn main:app --reload

Endpoints:
    POST /chat        →  Student Q&A chatbot
    POST /questions   →  OBE exam question generator
    GET  /            →  Health check
"""

from fastapi import FastAPI
from src.routers import chat, questions

app = FastAPI(
    title       = "ENA OBE Assistant",
    description = "RAG-powered chatbot and OBE question generator for Electrical Network Analysis",
    version     = "1.0.0",
)

# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(chat.router,      prefix="/chat",      tags=["Chatbot"])
app.include_router(questions.router, prefix="/questions", tags=["Question Generator"])

# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "ENA OBE Assistant is running."}
