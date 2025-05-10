# app/services/vector_store.py
from langchain.docstore.document import Document
import os
import logging
import json
import psycopg2
from datetime import datetime
from sqlalchemy import create_engine, text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
from sentence_transformers import SentenceTransformer
from app.config import (
    PG_CONNECTION_STRING,
    DOC_COLLECTION,
    CHAT_MEMORY_COLLECTION,
    EMBEDDING_MODEL,
    SENTENCE_MODEL,
    OPENAI_API_KEY,
    OPENAI_API_BASE,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
#DOC_PATH = r"C:\Users\praveen.singh1\Desktop\DataKonnect\documents\lmg_document.txt"
#RELATIONSHIP_CHUNKS_JSON = r"C:\Users\praveen.singh1\Desktop\DataKonnect\documents\relationship_chunks.json"
#DDL_INFORMATION = r"C:\Users\praveen.singh1\Desktop\DataKonnect\documents\ddl_information.json"'''
BASE_DIR = Path(__file__).resolve().parents[2]

DOC_PATH                = BASE_DIR / "documents" / "lmg_document.txt"
RELATIONSHIP_CHUNKS_JSON = BASE_DIR / "documents" / "relationship_chunks.json"
DDL_INFORMATION         = BASE_DIR / "documents" / "ddl_information.json"

print(DOC_PATH)

# ─── Initialize PGVector stores ────────────────────────────────────────────────
# Document/relationship chunks store
doc_vectorstore = PGVector(
    embedding_function=EMBEDDING_MODEL,
    collection_name=DOC_COLLECTION,
    connection_string=PG_CONNECTION_STRING
)
# Chat memory store (for RAG memory of past Q&A)
chat_memory_store = PGVector.from_documents(
    documents=[],                  # empty list
    embedding=EMBEDDING_MODEL,     # your embedding function
    collection_name=CHAT_MEMORY_COLLECTION,
    connection_string=PG_CONNECTION_STRING
)

# ─── Helper functions for context validation/refinement ─────────────────────────
def is_query_valid_for_context(prompt, available_columns):
    for col in available_columns.split(","):
        if col.strip().lower() in prompt.lower():
            return True
    return False


def extract_missing_column(prompt, all_columns_list):
    keywords = prompt.lower().split()
    all_columns_flat = ','.join(all_columns_list).lower()
    for word in keywords:
        if word not in all_columns_flat:
            return word
    return None


def extract_missing_table(prompt, all_tables_list):
    for table in all_tables_list:
        if table.lower() in prompt.lower():
            return None
    return all_tables_list[0] if all_tables_list else None


def refine_prompt_with_context(prompt, missing_column=None, missing_table=None):
    additions = []
    if missing_column:
        additions.append(f"include column '{missing_column}'")
    if missing_table:
        additions.append(f"use table '{missing_table}'")
    return prompt + " (" + ", ".join(additions) + ")" if additions else prompt

# ─── Relationship chunks embedding & table setup ───────────────────────────────
def embed_relationship_chunks():
    with open(RELATIONSHIP_CHUNKS_JSON, 'r', encoding='utf-8') as file:
        chunks = json.load(file)

    conn = psycopg2.connect(PG_CONNECTION_STRING.replace("postgresql+psycopg2", "postgresql"))
    cur = conn.cursor()
    cur.execute("""
        CREATE EXTENSION IF NOT EXISTS vector;
        DROP TABLE IF EXISTS table_chunks;
        CREATE TABLE table_chunks (
            id SERIAL PRIMARY KEY,
            table_name TEXT,
            columns TEXT,
            chunk TEXT,
            embedding VECTOR(384)
        );
    """)
    conn.commit()

    for row in chunks:
        emb = SENTENCE_MODEL.encode(row["chunk"]).tolist()
        cur.execute(
            "INSERT INTO table_chunks (table_name, chunk, columns, embedding) VALUES (%s, %s, %s, %s)",
            (row["table"], row["chunk"], row["columns"], emb)
        )
    conn.commit()
    conn.close()
    logger.info("✅ Relationship chunks embedded and stored.")

# ─── Semantic search over relationship chunks ─────────────────────────────────
def semantic_search(question, top_k=1):
    conn = psycopg2.connect(PG_CONNECTION_STRING.replace("postgresql+psycopg2", "postgresql"))
    cur = conn.cursor()
    embedding = SENTENCE_MODEL.encode(question).tolist()
    cur.execute(
        """
        SELECT table_name, chunk, columns, embedding <-> %s::vector AS similarity
        FROM table_chunks
        ORDER BY similarity ASC
        LIMIT %s;
        """, (embedding, top_k)
    )
    results = cur.fetchall()
    conn.close()
    return results

# ─── Build or load document vector store ────────────────────────────────────────
def build_document_store():
    embed_relationship_chunks()
    loader = TextLoader(DOC_PATH, encoding='utf-8')
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    doc_chunks = splitter.split_documents(documents)

    relationship_docs = [
        Document(page_content=chunk["chunk"], metadata={"table": chunk["table"]})
        for chunk in json.load(open(RELATIONSHIP_CHUNKS_JSON, 'r', encoding='utf-8'))
    ]
    combined = relationship_docs + doc_chunks

    logger.info("🔌 Storing combined chunks in PostgreSQL vector store...")
    return PGVector.from_documents(
        documents=combined,
        embedding=EMBEDDING_MODEL,
        collection_name=DOC_COLLECTION,
        connection_string=PG_CONNECTION_STRING
    )

# ─── Retriever initializer ────────────────────────────────────────────────────
def get_vectorstore_retriever():
    try:
        # If document store is not initialized, build it
        engine = create_engine(PG_CONNECTION_STRING)
        with engine.connect() as conn:
            res = conn.execute(text(
                "SELECT to_regclass('public.table_chunkss')"
            ))
            exists = res.scalar() is not None
        if exists:
            logger.info("✅ Using existing document embeddings.")
        else:
            logger.info("🔄 Creating new document embeddings.")
            build_document_store()

        return doc_vectorstore.as_retriever(search_kwargs={"k": 3})

    except Exception as e:
        logger.error(f"❌ Failed to initialize vector store: {e}")
        raise

# ─── Chat memory helpers ───────────────────────────────────────────────────────

def add_to_memory(user_question: str, bot_reply: str) -> None:
    """Embed and store a Q&A pair in Postgres."""
    try:
        doc = Document(
            page_content=f"Q: {user_question}\nA: {bot_reply}",
            metadata={"timestamp": datetime.utcnow().isoformat()}
        )
        chat_memory_store.add_documents([doc])
        logger.info("✅ Stored Q&A in chat memory.")
    except Exception as e:
        logger.error(f"❌ Failed to add to memory: {e}")
        raise


def fetch_memory(query: str, k: int = 3) -> str:
    """Retrieve the most similar past Q&A documents."""
    try:
        docs = chat_memory_store.similarity_search(query, k=k)
        logger.info(f"🔍 Retrieved {len(docs)} memory docs for query.")
        logger.info(docs)
        return "\n".join(d.page_content for d in docs)
    except Exception as e:
        logger.error(f"❌ Memory fetch failed: {e}")
        raise
   

# End of file


#add_to_memory("ping", "pong")


'''def run_with_feedback_loop(prompt, max_retries=1, top_k=3):
    """
    Executes semantic search with a feedback loop to resolve missing tables or columns.
    """
    tried_prompts = [prompt]
    retry_count = 0

    while retry_count <= max_retries:
        results = semantic_search(prompt, top_k=top_k)

        if not results:
            logger.warning(f"⚠️ No results for prompt: {prompt}")
            break

        for table_name, chunk, columns, similarity in results:
            if is_query_valid_for_context(prompt, columns):
                logger.info(f"✅ Found context match at retry {retry_count}")
                return {
                    "final_prompt": prompt,
                    "matched_table": table_name,
                    "matched_columns": columns,
                    "chunk": chunk,
                    "similarity": similarity
                }

        # If context mismatch, refine the prompt
        missing_col = extract_missing_column(prompt, [col for _, _, col, _ in results])
        missing_tbl = extract_missing_table(prompt, [tbl for tbl, _, _, _ in results])

        if missing_col or missing_tbl:
            prompt = refine_prompt_with_context(prompt, missing_col, missing_tbl)
            logger.info(f"🔁 Retrying with refined prompt: {prompt}")
            tried_prompts.append(prompt)
            retry_count += 1
        else:
            logger.warning("❌ No more refinements possible.")
            break

    logger.warning("🚫 Max retries exceeded or no resolution.")
    return None

if __name__ == "__main__":
    prompt = "What is the net sales and cost per location?"

    print("\n🔄 Running with Feedback Loop...\n")
    result = run_with_feedback_loop(prompt, max_retries=3)

    if result:
        print(f"✅ Final Prompt: {result['final_prompt']}")
        print(f"📘 Table: {result['matched_table']}")
        print(f"📊 Columns: {result['matched_columns']}")
        print(f"🔢 Similarity: {result['similarity']:.4f}")
        print(f"🧩 Chunk: {result['chunk']}")
    else:
        print("❌ Failed to resolve prompt.")'''

if __name__ == "__main__":
    # Example prompt to test semantic search
    prompt = "list the concept names which has maximum salse"

    # Build or load the vector store
    retriever = get_vectorstore_retriever()

    # Run semantic search using your custom function (optional)
    print("\n🔍 Running semantic search from table_chunks...\n")
    results = semantic_search(prompt, top_k=1)
    for table_name, chunk, columns, similarity in results:
        print(f"📘 Table: {table_name}\n🧩 Chunk: {chunk}\n📊 Columns: {columns}\n🔢 Similarity: {similarity:.4f}\n")

    # Run search via LangChain retriever (alternative method)
    print("\n🌐 Running LangChain vectorstore retriever search...\n")
    docs = retriever.get_relevant_documents(prompt)
    for doc in docs:
        print(f"📄 Content:\n{doc.page_content}\n📎 Metadata: {doc.metadata}\n" + "-" * 80)
