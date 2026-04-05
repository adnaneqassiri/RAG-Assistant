import os
from fastapi import FastAPI
from fastapi import APIRouter, UploadFile, File, HTTPException
from src.data_loader import process_file, split_documents
from src.embeddings import EmbManagaer
from src.vectorstore import VectorStore
from src.search import RAGRetriever
from src.model import GroqLLM, AdvancedRAGPipline
from src.query_router import simple_router
from src.not_rag import no_rag
from pydantic import BaseModel
from typing import List
from pathlib import Path


app = FastAPI(
    title="RAG Assistant API",
    version="1.1"
)


emb_manager = EmbManagaer("all-MiniLM-L6-v2")
vector_store = VectorStore("files", "data/vector_store")
retriever = RAGRetriever(vector_store, emb_manager)


router = APIRouter()


class QueryRequest(BaseModel):
    question: str




# ---------------------------
# QUERY ENDPOINT
# ---------------------------
@router.post("/query")
def query(req: QueryRequest):
    question = req.question
    result = simple_router(query)
    print(result)
    if ~result['retrieval_needed']:
        output = no_rag(question, result['task_type'], result['temperature'])
    else :
        llm = GroqLLM(model_name = "llama-3.3-70b-versatile", temp=result['temperature'])
        rag = AdvancedRAGPipline(retriever, llm)
        output = rag.query(question, 5, 2, False, False)
    
    return {
        "answer": output["answer"],
        "sources": output.get("sources", [])
    }


# ---------------------------
# UPLOAD PDF ENDPOINT
# ---------------------------
@router.post("/upload")
def upload_files(files: List[UploadFile] = File(...)):
    
    uploadd_dir = 'data/upload'
    os.makedirs(uploadd_dir, exist_ok=True)
    
    all_documents = []
    for file in files:
        filename = Path(file.filename).name
        save_path = f"{uploadd_dir}/{filename}"
        with open(save_path, "wb") as f:
            f.write(file.file.read())
        
        try:
            documents = process_file(save_path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"{filename}: {str(e)}")
        
        all_documents.extend(documents)
        
    if not all_documents:
        raise HTTPException(status_code=400, detail="No valid documents were processed.")
    chunks = split_documents(all_documents)
    texts = [doc.page_content for doc in chunks]
    embeddings = emb_manager.generate_embeddings(texts)

    vector_store.add_document(chunks, embeddings)
    print(vector_store.collection.count())
    return {
        "message": "Files processed successfully",
        "files_uploaded": len(files),
        "chunks_created": len(chunks)
    }


app.include_router(router)