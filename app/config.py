# app/config.py
import os
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ─── Database ─────────────────────────────────────────────────────────────────
PG_CONNECTION_STRING = os.getenv(
    "PGVECTOR_CONNECTION_STRING",
    "postgresql+psycopg2://myuser:mypassword@localhost:5433/mydatabase"
)

# ─── Collections ──────────────────────────────────────────────────────────────
DOC_COLLECTION = os.getenv("DOC_COLLECTION", "lmg_embeddings")
CHAT_MEMORY_COLLECTION = os.getenv("CHAT_MEMORY_COLLECTION", "chat_memory")

# ─── Embedding Models ─────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
SENTENCE_MODEL_NAME  = os.getenv("SENTENCE_MODEL_NAME",  "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
SENTENCE_MODEL = SentenceTransformer(SENTENCE_MODEL_NAME)

# ─── LLM Configuration ─────────────────────────────────────────────────────────
os.environ["OPENAI_API_KEY"] = "sk-GO1cPGBgEJhCbgOUSstpAA"
os.environ["OPENAI_API_BASE"] = "https://lmlitellm.landmarkgroup.com/"
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_API_BASE = os.environ["OPENAI_API_BASE"]

LLM_MODEL          = "landmark-gpt-4o-mini"
LLM_TEMPERATURE    = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS     = int(os.getenv("LLM_MAX_TOKENS", "1000"))


# Usage:
#  from app.config import (
#      PG_CONNECTION_STRING, DOC_COLLECTION, CHAT_MEMORY_COLLECTION,
#      EMBEDDING_MODEL, SENTENCE_MODEL,
#      OPENAI_API_KEY, OPENAI_API_BASE, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
#  )
