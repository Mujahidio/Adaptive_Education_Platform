# -*- coding: utf-8 -*-
import os
import io
import json
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import httpx
import PyPDF2
from dotenv import load_dotenv
import uvicorn
from datetime import datetime

# Load environment variables
load_dotenv()

app = FastAPI(title="PDF Processing API", version="1.0.0")

# In-memory storage for development (use database in production)
documents_storage = {}

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "http://localhost:3003",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:3002",
        "http://127.0.0.1:3003"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
async def ping():
    return {"message": "pong", "status": "success"}

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "deepseek/deepseek-r1-distill-llama-70b:free"

# Response models
class SummaryResponse(BaseModel):
    summary: str
    key_points: List[str]

class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    correct_answer: int
    explanation: str

class QuizResponse(BaseModel):
    questions: List[QuizQuestion]

class Flashcard(BaseModel):
    front: str
    back: str

class FlashcardsResponse(BaseModel):
    flashcards: List[Flashcard]

class ProcessingResponse(BaseModel):
    summary: SummaryResponse
    quiz: QuizResponse
    flashcards: FlashcardsResponse

class DocumentBase(BaseModel):
    title: str

class DocumentCreate(DocumentBase):
    pass

class Document(DocumentBase):
    id: str
    created_at: str

class DatedStudyData(BaseModel):
    date: str
    duration: int
    sessions: int

class DocumentPerformance(BaseModel):
    document_title: str
    accuracy: float
    attempts: int

class RecentQuizPerformance(BaseModel):
    date: str
    score: float
    quiz_title: str

class OverallAnalytics(BaseModel):
    total_study_time: int
    current_streak: int
    longest_streak: int
    total_flashcards_seen: int
    total_flashcards_mastered: int
    flashcard_accuracy_overall: float
    total_quizzes_completed: int
    average_quiz_score_overall: float
    study_sessions_this_week_count: int

class AnalyticsPageData(BaseModel):
    overall_analytics: OverallAnalytics
    study_sessions_chart_data: List[DatedStudyData]
    flashcard_performance_chart_data: List[DocumentPerformance]
    quiz_performance_chart_data: List[RecentQuizPerformance]

# Helper functions
def extract_text_from_pdf(pdf_file: UploadFile) -> str:
    """Extract text content from uploaded PDF file"""
    try:
        pdf_content = pdf_file.file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting PDF text: {str(e)}")

async def call_openrouter_api(prompt: str, max_tokens: int = 2000) -> str:
    """Make API call to OpenRouter with DeepSeek model"""
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OpenRouter API key not configured")
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",  # Replace with your domain
        "X-Title": "PDF Processing App"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=f"OpenRouter API error: {str(e)}")
        except KeyError as e:
            raise HTTPException(status_code=500, detail=f"Unexpected API response format: {str(e)}")

def create_summary_prompt(text: str) -> str:
    """Create prompt for summary generation"""
    return f"""
    Please analyze the following text and provide a comprehensive summary along with key points.
    
    Text to summarize:
    {text}
    
    Please respond in the following JSON format:
    {{
        "summary": "A comprehensive summary of the text (2-3 paragraphs)",
        "key_points": ["Key point 1", "Key point 2", "Key point 3", "Key point 4", "Key point 5"]
    }}
    
    Ensure the response is valid JSON.
    """

def create_quiz_prompt(text: str) -> str:
    """Create prompt for quiz generation"""
    return f"""
    Based on the following text, create 5 multiple-choice questions to test understanding.
    
    Text:
    {text}
    
    Please respond in the following JSON format:
    {{
        "questions": [
            {{
                "question": "Question text here?",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "correct_answer": 0,
                "explanation": "Brief explanation of why this is correct"
            }}
        ]
    }}
    
    Make sure:
    - Each question has exactly 4 options
    - correct_answer is the index (0-3) of the correct option
    - Questions test different aspects of the content
    - Ensure the response is valid JSON
    """

def create_flashcards_prompt(text: str) -> str:
    """Create prompt for flashcard generation"""
    return f"""
    Based on the following text, create 8 flashcards for studying key concepts.
    
    Text:
    {text}
    
    Please respond in the following JSON format:
    {{
        "flashcards": [
            {{
                "front": "Question or concept",
                "back": "Answer or explanation"
            }}
        ]
    }}
    
    Make sure:
    - Each flashcard tests an important concept
    - Front side is concise (question/term)
    - Back side provides clear explanation
    - Cover different topics from the text
    - Ensure the response is valid JSON
    """

def parse_json_response(response: str) -> Dict[str, Any]:
    """Safely parse JSON response from API"""
    try:
        # Try to find JSON in the response
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON found in response")
        
        json_str = response[start_idx:end_idx]
        return json.loads(json_str)
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"Error parsing API response: {str(e)}")

# API Endpoints
@app.post("/upload-pdf", response_model=ProcessingResponse)
async def upload_and_process_pdf(file: UploadFile = File(...)):
    """Upload PDF and generate summary, quiz, and flashcards"""
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Extract text from PDF
    pdf_text = extract_text_from_pdf(file)
    
    if not pdf_text.strip():
        raise HTTPException(status_code=400, detail="No text found in PDF")
    
    try:
        # Generate summary
        summary_prompt = create_summary_prompt(pdf_text)
        summary_response = await call_openrouter_api(summary_prompt)
        summary_data = parse_json_response(summary_response)
        
        # Generate quiz
        quiz_prompt = create_quiz_prompt(pdf_text)
        quiz_response = await call_openrouter_api(quiz_prompt)
        quiz_data = parse_json_response(quiz_response)
        
        # Generate flashcards
        flashcards_prompt = create_flashcards_prompt(pdf_text)
        flashcards_response = await call_openrouter_api(flashcards_prompt)
        flashcards_data = parse_json_response(flashcards_response)
        
        # Construct response
        return ProcessingResponse(
            summary=SummaryResponse(**summary_data),
            quiz=QuizResponse(**quiz_data),
            flashcards=FlashcardsResponse(**flashcards_data)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/generate-summary", response_model=SummaryResponse)
async def generate_summary(text: str):
    """Generate summary from provided text"""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        prompt = create_summary_prompt(text)
        response = await call_openrouter_api(prompt)
        data = parse_json_response(response)
        return SummaryResponse(**data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")

@app.post("/generate-quiz", response_model=QuizResponse)
async def generate_quiz(text: str):
    """Generate quiz from provided text"""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        prompt = create_quiz_prompt(text)
        response = await call_openrouter_api(prompt)
        data = parse_json_response(response)
        return QuizResponse(**data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating quiz: {str(e)}")

@app.post("/generate-flashcards", response_model=FlashcardsResponse)
async def generate_flashcards(text: str):
    """Generate flashcards from provided text"""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        prompt = create_flashcards_prompt(text)
        response = await call_openrouter_api(prompt)
        data = parse_json_response(response)
        return FlashcardsResponse(**data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating flashcards: {str(e)}")

@app.post("/documents/upload", response_model=Document)
async def upload_document(pdf: UploadFile = File(...), title: str = Form(...)):
    """Upload a PDF document"""
    if not pdf.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        # Extract text from PDF to validate it
        pdf_text = extract_text_from_pdf(pdf)
        
        if not pdf_text.strip():
            raise HTTPException(status_code=400, detail="No text found in PDF")
        
        # Generate a unique ID for the document
        doc_id = str(int(datetime.now().timestamp()))
        
        # Store document data and PDF text for processing
        documents_storage[doc_id] = {
            "id": doc_id,
            "title": title,
            "created_at": datetime.now().isoformat(),
            "user_id": "default-user-id",
            "file_path": f"/uploads/{doc_id}.pdf",
            "pdf_text": pdf_text,
            "processed": False
        }
        
        # Create document response
        document = Document(
            id=doc_id,
            title=title,
            created_at=datetime.now().isoformat()
        )
        
        return document
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@app.post("/documents/{document_id}/process")
async def process_document(document_id: str):
    """Process an uploaded document"""
    try:
        if document_id not in documents_storage:
            raise HTTPException(status_code=404, detail="Document not found")
        
        document_data = documents_storage[document_id]
        pdf_text = document_data.get("pdf_text", "")
        
        if not pdf_text:
            raise HTTPException(status_code=400, detail="No text content found for this document")
        
        # Generate summary using AI
        summary_prompt = create_summary_prompt(pdf_text)
        summary_response = await call_openrouter_api(summary_prompt)
        summary_data = parse_json_response(summary_response)
        
        # Generate quiz using AI
        quiz_prompt = create_quiz_prompt(pdf_text)
        quiz_response = await call_openrouter_api(quiz_prompt)
        quiz_data = parse_json_response(quiz_response)
        
        # Generate flashcards using AI
        flashcards_prompt = create_flashcards_prompt(pdf_text)
        flashcards_response = await call_openrouter_api(flashcards_prompt)
        flashcards_data = parse_json_response(flashcards_response)
        
        # Transform the AI responses to match frontend schema
        summary = {
            "id": f"sum-{document_id}",
            "document_id": document_id,
            "content": summary_data.get("summary", "Summary not available"),
            "created_at": datetime.now().isoformat()
        }
        
        # Transform flashcards from AI response
        flashcards = []
        for i, card in enumerate(flashcards_data.get("flashcards", [])):
            flashcards.append({
                "id": f"fc-{document_id}-{i+1}",
                "document_id": document_id,
                "question": card.get("front", ""),
                "answer": card.get("back", ""),
                "created_at": datetime.now().isoformat()
            })
        
        # Transform quiz from AI response
        quiz_questions = []
        for i, q in enumerate(quiz_data.get("questions", [])):
            quiz_questions.append({
                "id": f"q-{document_id}-{i+1}",
                "quiz_id": f"quiz-{document_id}",
                "question": q.get("question", ""),
                "options": q.get("options", []),
                "correct_answer": q.get("options", [""])[q.get("correct_answer", 0)] if q.get("options") else "",
                "created_at": datetime.now().isoformat()
            })
        
        quiz = {
            "id": f"quiz-{document_id}",
            "document_id": document_id,
            "title": f"Quiz: {document_data['title']}",
            "created_at": datetime.now().isoformat(),
            "questions": quiz_questions
        }
        
        # Update stored document with processed content
        documents_storage[document_id].update({
            "summary": summary,
            "flashcards": flashcards,
            "quiz": quiz,
            "processed": True
        })
        
        return {
            "status": "success",
            "message": "Document processed successfully with AI-generated content",
            "document_id": document_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": MODEL_NAME}

@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify backend-frontend communication"""
    return {"status": "ok", "message": "Backend is connected successfully"}

@app.get("/documents")
async def get_documents():
    """Get a list of processed documents"""
    try:
        # Return list of documents from storage
        documents = []
        for doc_id, doc_data in documents_storage.items():
            documents.append({
                "id": doc_data["id"],
                "title": doc_data["title"],
                "created_at": doc_data["created_at"]
            })
        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching documents: {str(e)}")

@app.get("/documents/{document_id}")
async def get_document(document_id: str):
    """Get document details by ID"""
    try:
        # Check if document exists in storage
        if document_id in documents_storage:
            doc_data = documents_storage[document_id]
            
            # If document has been processed, return the real data
            if doc_data.get("processed", False):
                return {
                    "id": doc_data["id"],
                    "title": doc_data["title"],
                    "created_at": doc_data["created_at"],
                    "user_id": doc_data["user_id"],
                    "file_path": doc_data["file_path"],
                    "summary": doc_data.get("summary"),
                    "flashcards": doc_data.get("flashcards", []),
                    "quiz": doc_data.get("quiz")
                }
            else:
                # Document exists but not processed yet, return basic info
                return {
                    "id": doc_data["id"],
                    "title": doc_data["title"],
                    "created_at": doc_data["created_at"],
                    "user_id": doc_data["user_id"],
                    "file_path": doc_data["file_path"],
                    "summary": None,
                    "flashcards": [],
                    "quiz": None
                }
        
        # Document not found, return mock data for demo purposes
        document = {
            "id": document_id,
            "title": "Sample Document: AI Fundamentals",
            "created_at": datetime.now().isoformat(),
            "user_id": "default-user-id",
            "file_path": f"/uploads/{document_id}.pdf",
            "summary": {
                "id": f"sum-{document_id}",
                "document_id": document_id,
                "content": "This document provides a comprehensive introduction to Artificial Intelligence, covering key concepts such as machine learning, neural networks, natural language processing, and computer vision. It explores the historical development of AI, current applications across various industries, and future prospects for AI technology.",
                "created_at": datetime.now().isoformat()
            },
            "flashcards": [
                {
                    "id": f"fc-{document_id}-1",
                    "document_id": document_id,
                    "question": "What is Artificial Intelligence?",
                    "answer": "Artificial Intelligence is the simulation of human intelligence processes by machines, especially computer systems.",
                    "created_at": datetime.now().isoformat()
                },
                {
                    "id": f"fc-{document_id}-2",
                    "document_id": document_id,
                    "question": "What are the main types of machine learning?",
                    "answer": "Supervised learning, unsupervised learning, and reinforcement learning.",
                    "created_at": datetime.now().isoformat()
                },
                {
                    "id": f"fc-{document_id}-3",
                    "document_id": document_id,
                    "question": "What is a neural network?",
                    "answer": "A computing system inspired by biological neural networks that processes information using interconnected nodes.",
                    "created_at": datetime.now().isoformat()
                }
            ],
            "quiz": {
                "id": f"quiz-{document_id}",
                "document_id": document_id,
                "title": "AI Fundamentals Quiz",
                "created_at": datetime.now().isoformat(),
                "questions": [
                    {
                        "id": f"q-{document_id}-1",
                        "quiz_id": f"quiz-{document_id}",
                        "question": "Which of the following is a subset of AI focused on learning from data?",
                        "options": ["Machine Learning", "Computer Graphics", "Database Management", "Web Development"],
                        "correct_answer": "Machine Learning",
                        "created_at": datetime.now().isoformat()
                    },
                    {
                        "id": f"q-{document_id}-2",
                        "quiz_id": f"quiz-{document_id}",
                        "question": "What type of AI can perform any intellectual task that a human can do?",
                        "options": ["Narrow AI", "General AI", "Super AI", "Weak AI"],
                        "correct_answer": "General AI",
                        "created_at": datetime.now().isoformat()
                    }
                ]
            }
        }
        return document
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Document not found: {str(e)}")

# Analytics endpoints
@app.get("/analytics/pagedata", response_model=AnalyticsPageData)
async def get_analytics_pagedata():
    """Get analytics data for the analytics page"""
    # TODO: In a real application, this would fetch from a database
    # For now, return mock data for development/testing
    mock_data = {
        "overall_analytics": {
            "total_study_time": 3600,  # 1 hour in seconds
            "current_streak": 3,
            "longest_streak": 5,
            "total_flashcards_seen": 50,
            "total_flashcards_mastered": 30,
            "flashcard_accuracy_overall": 75.0,
            "total_quizzes_completed": 10,
            "average_quiz_score_overall": 85.0,
            "study_sessions_this_week_count": 5
        },
        "study_sessions_chart_data": [
            {"date": "2025-06-06", "duration": 30, "sessions": 1},
            {"date": "2025-06-07", "duration": 45, "sessions": 2},
            {"date": "2025-06-08", "duration": 60, "sessions": 2},
            {"date": "2025-06-09", "duration": 30, "sessions": 1},
            {"date": "2025-06-10", "duration": 90, "sessions": 3},
            {"date": "2025-06-11", "duration": 60, "sessions": 2},
            {"date": "2025-06-12", "duration": 45, "sessions": 2}
        ],
        "flashcard_performance_chart_data": [
            {"document_title": "Introduction to AI", "accuracy": 85.0, "attempts": 20},
            {"document_title": "Machine Learning Basics", "accuracy": 75.0, "attempts": 15},
            {"document_title": "Neural Networks", "accuracy": 70.0, "attempts": 10}
        ],
        "quiz_performance_chart_data": [
            {"date": "2025-06-01", "score": 75.0, "quiz_title": "AI Quiz 1"},
            {"date": "2025-06-03", "score": 80.0, "quiz_title": "ML Quiz"},
            {"date": "2025-06-06", "score": 85.0, "quiz_title": "NN Quiz"},
            {"date": "2025-06-09", "score": 90.0, "quiz_title": "AI Quiz 2"},
            {"date": "2025-06-12", "score": 95.0, "quiz_title": "Final Quiz"}        ]
    }
    return mock_data

# Session and analytics tracking endpoints
@app.post("/analytics/session/start")
async def start_study_session(session_data: dict):
    """Start a new study session"""
    try:
        # In a real app, this would save to database
        # For now, just return a mock session ID
        session_id = f"session-{int(datetime.now().timestamp())}"
        return {"id": session_id, "status": "started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting session: {str(e)}")

@app.post("/analytics/session/{session_id}/end")
async def end_study_session(session_id: str):
    """End a study session"""
    try:
        # In a real app, this would update the session in database
        return {"status": "ended", "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ending session: {str(e)}")

@app.post("/analytics/flashcard/attempt")
async def track_flashcard_attempt(attempt_data: dict):
    """Track a flashcard attempt"""
    try:
        # In a real app, this would save the attempt to database
        return {"status": "tracked", "attempt_id": f"attempt-{int(datetime.now().timestamp())}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error tracking flashcard attempt: {str(e)}")

@app.post("/analytics/quiz/attempt")
async def track_quiz_attempt(quiz_data: dict):
    """Track a quiz attempt"""
    try:
        # In a real app, this would save the quiz results to database
        return {"status": "tracked", "quiz_attempt_id": f"quiz-attempt-{int(datetime.now().timestamp())}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error tracking quiz attempt: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)