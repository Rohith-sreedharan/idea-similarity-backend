from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn
import numpy as np

# Initialize FastAPI app
app = FastAPI(
    title="FastAPI Backend",
    description="A FastAPI backend with ML capabilities",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample project ideas
project_ideas = [
    "Build a machine learning model to predict house prices based on location and features",
    "Create a web application for real-time chat using WebSockets and React",
    "Develop an AI-powered recommendation system for e-commerce products"
]

# Encode project ideas into embeddings
project_embeddings = model.encode(project_ideas)

# Pydantic models for request/response
class HealthResponse(BaseModel):
    status: str
    message: str

class CheckInput(BaseModel):
    text: str

class CheckResponse(BaseModel):
    status: str
    similarity_score: float
    most_similar_idea: str

# Routes
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return HealthResponse(
        status="success",
        message="FastAPI backend is running!"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="All systems operational"
    )

@app.post("/check", response_model=CheckResponse)
async def check_similarity(input_data: CheckInput):
    """
    Check similarity between input text and sample project ideas
    Returns status, similarity score, and most similar idea if similarity > 0.85
    """
    # Encode the input text
    input_embedding = model.encode([input_data.text])
    
    # Compute cosine similarity with all project ideas
    similarities = cosine_similarity(input_embedding, project_embeddings)[0]
    
    # Find the most similar idea
    max_similarity_idx = np.argmax(similarities)
    max_similarity_score = float(similarities[max_similarity_idx])
    most_similar_idea = project_ideas[max_similarity_idx]
    
    # Check if similarity exceeds threshold
    if max_similarity_score > 0.85:
        return CheckResponse(
            status="high_similarity",
            similarity_score=max_similarity_score,
            most_similar_idea=most_similar_idea
        )
    else:
        return CheckResponse(
            status="low_similarity",
            similarity_score=max_similarity_score,
            most_similar_idea=most_similar_idea
        )

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Enable auto-reload during development
    )
