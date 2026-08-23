import os
import json
from groq import Groq
from fastapi import APIRouter
from pydantic import BaseModel
from src.pipeline import retrieve, build_context

router = APIRouter()
import os
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class QuestionRequest(BaseModel):
    clo:           str
    bloom:         str
    q_type:        str
    num_questions: int
    marks:         int
    topic:         str = ""

class QuestionResponse(BaseModel):
    questions: list[dict]
    sources:   list[str]

@router.post("/", response_model=QuestionResponse)
def generate_questions(request: QuestionRequest):
    query   = request.topic if request.topic else request.clo
    chunks  = retrieve(query)
    context = build_context(chunks)
    sources = list({c["source"] for c in chunks})

    prompt = f"""You are an OBE exam question generator for Electrical Network Analysis (ENA).

Course material context:
{context}

Generate exactly {request.num_questions} question(s):
- CLO: {request.clo}
- Bloom's Level: {request.bloom}
- Question Type: {request.q_type}
- Marks: {request.marks}
- Topic: {request.topic if request.topic else "As per CLO"}

Rules:
- For MCQ: 4 options (A-D), mark correct answer
- For Short: 3-5 line expected answer
- For Long: numerical/analytical problem
- Return ONLY valid JSON array, no markdown

JSON format:
[
  {{
    "q_no": 1,
    "clo": "CLO1",
    "bloom": "Apply (C3)",
    "type": "MCQ",
    "marks": 5,
    "question": "Question text here",
    "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
    "answer": "A"
  }}
]
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    questions = json.loads(raw)

    return QuestionResponse(questions=questions, sources=sources)