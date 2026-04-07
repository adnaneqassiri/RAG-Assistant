import os
from fastapi import FastAPI
from fastapi import APIRouter, UploadFile, File, HTTPException
from src.data_loader import process_file, split_documents
from src.embeddings import EmbManagaer
from src.vectorstore import VectorStore
from src.model_v2 import AdvancedRAGPipline
from src.query_router import retrieval_router
from src.not_rag import no_rag
from pydantic import BaseModel
from typing import List
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "upload"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

app = FastAPI(
    title="RAG Assistant API",
    version="1.1"
)


emb_manager = EmbManagaer("all-MiniLM-L6-v2")
vector_store = VectorStore("files", str(VECTOR_STORE_DIR))
rag = AdvancedRAGPipline("llama-3.3-70b-versatile" , vector_store, emb_manager)

router = APIRouter()


class QueryRequest(BaseModel):
    question: str


# Router returns 
# {
#  'retrieval_needed': True,
#  'search_query': 'ESRGAN training SISR',
#  'generation_query': 'How is ESRGAN trained in the context of SISR',
#  'metadata_candidates': {
#       'course': '',
#       'doc_type': 'SISR document',
#       'sheet_number': '',
#       'question_number': ''
#       },
#  'metadata_confidence': {
#       'course': 0.0,
#       'doc_type': 0.8,
#       'sheet_number': 0.0,
#       'question_number': 0.0
#       },
#  'search_profile': 'keyword_heavy',
#  'rule_triggered': True,
#  'decision_source': 'rule_override'
# }


# ---------------------------
# QUERY ENDPOINT
# ---------------------------
@router.post("/query")
def query(req: QueryRequest):
    question = req.question
    router = retrieval_router(question)
    retrieval_needed = router["retrieval_needed"]
    search_query = router["search_query"] or question
    generation_query = router["generation_query"] or question
    search_profile = router["search_profile"]

    # Metadata candidates
    meta = router["metadata_candidates"]

    # Metadata confidence
    conf = router["metadata_confidence"]
    print(router)
    
    if not retrieval_needed:
        output = no_rag(generation_query, history=rag.history)
        rag.history.append({
            "question": question,
            "answer": output["answer"],
            "sources": output.get("sources", []),
        })
    else:
        output = rag.query(
            query=question,
            generation_query=generation_query,
            search_query=search_query,
            search_profile=search_profile,
            metadata=meta,
            metadata_confidence=conf,
        )
    
    return {
        "answer": output["answer"],
        "sources": output.get("sources", [])
    }


# ---------------------------
# UPLOAD PDF ENDPOINT
# ---------------------------
@router.post("/upload")
def upload_files(files: List[UploadFile] = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    
    all_documents = []
    skipped_files = []
    for file in files:
        filename = Path(file.filename).name
        save_path = UPLOAD_DIR / filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())
        
        try:
            documents = process_file(str(save_path))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"{filename}: {str(e)}")

        file_hash = documents[0].metadata.get("file_hash") if documents else None
        if file_hash and vector_store.document_exists(file_hash=file_hash):
            skipped_files.append(filename)
            continue
        
        all_documents.extend(documents)
        
    if not all_documents:
        return {
            "message": "No new files were indexed",
            "files_uploaded": len(files),
            "files_skipped": skipped_files,
            "chunks_created": 0
        }
    chunks = split_documents(all_documents)
    texts = [doc.page_content for doc in chunks]
    embeddings = emb_manager.generate_embeddings(texts)

    vector_store.add_document(chunks, embeddings)
    print(vector_store.collection.count())
    return {
        "message": "Files processed successfully",
        "files_uploaded": len(files),
        "files_skipped": skipped_files,
        "chunks_created": len(chunks),
        "vector_store_path": str(VECTOR_STORE_DIR),
        "documents_in_collection": vector_store.collection.count(),
    }


app.include_router(router)
