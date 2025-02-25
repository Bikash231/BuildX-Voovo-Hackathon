# app/main.py

import os
import io
import json
import numpy as np
import pdfplumber
import hnswlib  # New import for hnswlib
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

from .database import Base, engine, get_db
from .models import Chunk, Quiz

##############################################################################
# CONFIGURATION & GLOBAL INITIALIZATION
##############################################################################

app = FastAPI()

# Allow CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create database tables
Base.metadata.create_all(bind=engine)

# Global variables for hnswlib and models
global_index = None  # Our hnswlib index
global_next_id = 0   # Counter for unique IDs for each vector
embedding_dim = 384  # Dimension for all-MiniLM-L6-v2 embeddings

# Load free embedding model (Sentence Transformers)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load free LLM model (Llama 2 7B Chat)
# Note: Running Llama 2 7B Chat requires substantial resources.
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", trust_remote_code=True)
llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", 
                                                   device_map="auto", 
                                                   trust_remote_code=True)

##############################################################################
# HELPER FUNCTIONS USING HNSWLIB
##############################################################################

def get_embedding(text: str) -> np.ndarray:
    """Compute embedding using Sentence Transformers (all-MiniLM-L6-v2)."""
    emb = embedding_model.encode(text)
    return np.array(emb, dtype=np.float32)

def chunk_text(text: str, chunk_size: int = 400) -> List[str]:
    """Splits text into chunks of approximately `chunk_size` words."""
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def create_hnsw_index():
    """Create a new hnswlib index with the specified embedding dimension."""
    index = hnswlib.Index(space='l2', dim=embedding_dim)
    index.init_index(max_elements=10000, ef_construction=200, M=16)
    index.set_ef(50)
    return index

def add_vector_to_index(index, vector: np.ndarray):
    """
    Adds a single vector to the hnswlib index.
    Returns the new unique ID for the vector.
    """
    global global_next_id
    current_id = global_next_id
    index.add_items(np.array([vector]), np.array([current_id]))
    global_next_id += 1
    return current_id

def retrieve_top_chunks(query: str, index, chunks: List[Chunk], top_k: int = 3):
    """Retrieve top_k chunks relevant to the query using hnswlib."""
    query_emb = get_embedding(query)
    labels, distances = index.knn_query(np.array([query_emb]), k=top_k)
    top_indices = labels[0].tolist()
    results = []
    for idx in top_indices:
        for c in chunks:
            if c.embedding_id == idx:
                results.append(c)
                break
    return results

def generate_quiz_with_llama2(context_text: str, subtopic: str, num_questions=6):
    """
    Uses Llama 2 to generate multiple-choice quiz questions.
    The output is expected to be in JSON format.
    """
    prompt = f"""
You are an AI that generates multiple-choice revision quizzes based on the provided context.
Context: \"\"\"{context_text}\"\"\"

Please create {num_questions} quiz questions covering the subtopic: {subtopic}.
Each question should have:
- 1 question text
- 1 correct answer
- 4 distractors

Return the result in JSON format with the structure:
[
  {{
    "question": "...",
    "correct_answer": "...",
    "distractors": ["...", "...", "...", "..."]
  }},
  ...
]
    """
    # Encode prompt and generate output
    input_ids = llama_tokenizer(prompt, return_tensors="pt").input_ids.to(llama_model.device)
    output_ids = llama_model.generate(input_ids, max_new_tokens=1024, do_sample=True, temperature=0.7)
    output = llama_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output

##############################################################################
# FASTAPI ENDPOINTS
##############################################################################

@app.on_event("startup")
def startup_event():
    """Initialize the global hnswlib index on startup."""
    global global_index
    global_index = create_hnsw_index()

@app.post("/upload")
def upload_file(subtopic: str = Form(...),
                file: UploadFile = File(...),
                db: Session = Depends(get_db)):
    """
    Upload a PDF, extract text, chunk it, compute embeddings,
    and store chunks in the database and hnswlib index.
    """
    global global_index
    file_name = file.filename

    # Read PDF file
    pdf_bytes = file.file.read()
    file.file.close()

    text_content = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_content += page_text + "\n"

    # Chunk the text
    chunked_texts = chunk_text(text_content, chunk_size=400)

    # Process each chunk: compute embedding and add to hnswlib & DB
    for chunk_str in chunked_texts:
        emb = get_embedding(chunk_str)
        new_id = add_vector_to_index(global_index, emb)

        db_chunk = Chunk(
            file_name=file_name,
            chunk_text=chunk_str,
            embedding_id=new_id,
            subtopic=subtopic
        )
        db.add(db_chunk)

    db.commit()
    return {"status": "ok", "message": f"File '{file_name}' uploaded and processed."}

class QuizRequest(BaseModel):
    subtopic: str
    num_questions: int = 6

@app.post("/generate-quizzes")
def generate_quizzes(request: QuizRequest, db: Session = Depends(get_db)):
    """
    Retrieve top chunks for the subtopic, generate quizzes using Llama 2,
    store the quizzes in the database, and return them.
    """
    global global_index

    # Retrieve all chunks for the subtopic from DB
    subtopic_chunks = db.query(Chunk).filter(Chunk.subtopic == request.subtopic).all()
    if not subtopic_chunks:
        return {"error": "No chunks found for this subtopic. Please upload materials first."}

    # Retrieve top chunks using hnswlib (using the subtopic as query)
    top_chunks = retrieve_top_chunks(request.subtopic, global_index, subtopic_chunks, top_k=3)
    combined_context = "\n".join([chunk.chunk_text for chunk in top_chunks])

    # Generate quizzes using Llama 2
    quiz_json_str = generate_quiz_with_llama2(combined_context, request.subtopic, request.num_questions)

    try:
        quiz_data = json.loads(quiz_json_str)
    except json.JSONDecodeError:
        return {"error": "Failed to parse quiz JSON from Llama 2.", "content": quiz_json_str}

    saved_quizzes = []
    for q in quiz_data:
        question_text = q.get("question", "")
        correct_answer = q.get("correct_answer", "")
        distractors = q.get("distractors", [])
        quiz_obj = Quiz(
            subtopic=request.subtopic,
            question=question_text,
            correct_answer=correct_answer,
            distractors="|".join(distractors)
        )
        db.add(quiz_obj)
        db.commit()
        db.refresh(quiz_obj)
        saved_quizzes.append({
            "id": quiz_obj.id,
            "question": quiz_obj.question,
            "correct_answer": quiz_obj.correct_answer,
            "distractors": distractors
        })

    return {"status": "ok", "quizzes": saved_quizzes}

@app.get("/quizzes")
def get_quizzes(subtopic: str, db: Session = Depends(get_db)):
    """
    Fetch quizzes for a given subtopic.
    """
    quiz_list = db.query(Quiz).filter(Quiz.subtopic == subtopic).all()
    output = []
    for q in quiz_list:
        distractors = q.distractors.split("|") if q.distractors else []
        output.append({
            "id": q.id,
            "question": q.question,
            "correct_answer": q.correct_answer,
            "distractors": distractors
        })
    return {"subtopic": subtopic, "quizzes": output}
