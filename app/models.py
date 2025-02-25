# app/models.py

from sqlalchemy import Column, Integer, String, Text
from .database import Base

class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, index=True)
    chunk_text = Column(Text)
    embedding_id = Column(Integer, index=True)  # FAISS index reference
    subtopic = Column(String, nullable=True)

class Quiz(Base):
    __tablename__ = "quizzes"
    id = Column(Integer, primary_key=True, index=True)
    subtopic = Column(String, index=True)
    question = Column(Text)
    correct_answer = Column(Text)
    distractors = Column(Text)  # Stored as pipe-separated string
