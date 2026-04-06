import hashlib
import re
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import numpy as np


def _compute_file_hash(file_path: str) -> str:
    with open(file_path, "rb") as file_handle:
        return hashlib.md5(file_handle.read()).hexdigest()


def _infer_filename_metadata(file_path: str) -> dict:
    path = Path(file_path)
    stem = path.stem
    normalized_stem = re.sub(r"[_\-]+", " ", stem).strip()
    lowered_stem = normalized_stem.lower()

    doc_type = ""
    if re.search(r"\b(td|tutorial)\b", lowered_stem):
        doc_type = "td"
    elif re.search(r"\b(tp|lab)\b", lowered_stem):
        doc_type = "tp"
    elif re.search(r"\b(cours|course|lecture)\b", lowered_stem):
        doc_type = "course"
    elif re.search(r"\b(exam|quiz|midterm|final)\b", lowered_stem):
        doc_type = "exam"

    sheet_number_match = re.search(r"\b(?:td|tp|sheet|chapter|chapitre)\s*[_\- ]?(\d+)\b", lowered_stem)
    question_number_match = re.search(r"\b(?:question|q)\s*[_\- ]?(\d+)\b", lowered_stem)

    return {
        "source_file": path.name,
        "source_file_lower": path.name.lower(),
        "source_stem": normalized_stem,
        "source_stem_lower": normalized_stem.lower(),
        "doc_type": doc_type,
        "sheet_number": sheet_number_match.group(1) if sheet_number_match else "",
        "question_number": question_number_match.group(1) if question_number_match else "",
    }


def process_file(file_path: str):
    ext = Path(file_path).suffix.lower()
    if ext == '.docx':
        loader = Docx2txtLoader(file_path)
    elif ext == '.pdf':
        loader = PyPDFLoader(file_path)
    elif ext == '.txt':
        loader = TextLoader(file_path, encoding='utf-8')
    else : 
        raise ValueError("Unsupported file type")
    
    documents = loader.load()
    file_hash = _compute_file_hash(file_path)
    shared_metadata = _infer_filename_metadata(file_path)
    shared_metadata["file_hash"] = file_hash
    shared_metadata["file_type"] = ext.replace('.', '')

    for doc in documents:
        doc.metadata.update(shared_metadata)
    
    return documents
    
def process_all_documents(pdf_dir):
    all_documents = []
    pdf_directory = Path(pdf_dir)
    
    pdf_files = list(pdf_directory.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} pdf file to process")
    
    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            
            file_hash = _compute_file_hash(str(pdf_file))
            shared_metadata = _infer_filename_metadata(str(pdf_file))
            shared_metadata["file_hash"] = file_hash
            shared_metadata["file_type"] = "pdf"

            for doc in documents:
                doc.metadata.update(shared_metadata)
                
            all_documents.extend(documents)
            print(f"Loaded {len(documents)} page")
            
        except Exception as e:
            print(f"Error: {e}")
        
    print(f"Total loaded documents: {len(all_documents)}")
    return all_documents


def split_documents(documents, chunk_size = 1000, overlap_size = 100) -> np.array:
    """
    function for spliting documents for better optimization for a RAG system
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap_size,
        length_function = len,
        separators=['\n\n', '\n', ' ', '']
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

    for chunk_index, chunk in enumerate(split_docs):
        chunk.metadata["chunk_index"] = chunk_index
    
    if split_docs:
        print(f"\nExample chunk:")
        print(f"\t1. Content: {split_docs[0].page_content[:100]}")
        print(f"\t2. Metadata: {split_docs[0].metadata}")
    
    return split_docs
