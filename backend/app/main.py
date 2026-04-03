import os
from fastapi import FastAPI
from fastapi import APIRouter, UploadFile, File
from src.data_loader import process_all_documents, split_documents
from src.embeddings import EmbManagaer
from src.vectorstore import VectorStore
from src.search import RAGRetriever
from src.model import GroqLLM, AdvancedRAGPipline
from pydantic import BaseModel


app = FastAPI(
    title="RAG Assistant API",
    version="1.0"
)

def get_rag():
    emb_manager = EmbManagaer("all-MiniLM-L6-v2")
    vector_store = VectorStore("pdfs", "data/vector_store")
    retriever = RAGRetriever(vector_store, emb_manager)
    llm = GroqLLM()
    return AdvancedRAGPipline(retriever, llm)

router = APIRouter()

# --- INIT (simple version) ---
emb_manager = EmbManagaer("all-MiniLM-L6-v2")
vector_store = VectorStore("pdfs", "data/vector_store")
retriever = RAGRetriever(vector_store, emb_manager)
llm = GroqLLM()
rag = AdvancedRAGPipline(retriever, llm)


class QueryRequest(BaseModel):
    Question: str




# ---------------------------
# QUERY ENDPOINT
# ---------------------------
@router.post("/query")
def query(req: QueryRequest):
    question = req.Question
    rag = get_rag()
    result = rag.query(question)

    return {
        "answer": result["answer"],
        "sources": result.get("sources", [])
    }


# ---------------------------
# UPLOAD PDF ENDPOINT
# ---------------------------
@router.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    save_path = f"data/pdf_files/{file.filename}"

    os.makedirs("data/pdf_files", exist_ok=True)

    with open(save_path, "wb") as f:
        f.write(file.file.read())

    docs = process_all_documents("data/pdf_files")
    chunks = split_documents(docs)

    texts = [d.page_content for d in chunks]
    embeddings = emb_manager.generate_embeddings(texts)

    vector_store.add_document(chunks, embeddings)
    print(vector_store.collection.count())
    return {
        "message": "PDF processed successfully"
    }


app.include_router(router)