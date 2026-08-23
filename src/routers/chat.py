import os
from groq import Groq
from fastapi import APIRouter
from pydantic import BaseModel
from src.pipeline import retrieve, build_context

router = APIRouter()

import os
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer:  str
    sources: list[str]

@router.post("/", response_model=ChatResponse)
def chat(request: ChatRequest):
    chunks  = retrieve(request.question)
    context = build_context(chunks)
    sources = list({c["source"] for c in chunks})

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are an ENA course assistant. Answer using ONLY the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.question}"}
        ]
    )
    return ChatResponse(answer=response.choices[0].message.content, sources=sources)