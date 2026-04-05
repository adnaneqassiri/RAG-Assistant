import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import numpy as np


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
    for doc in documents:
        doc.metadata['source_file'] = Path(file_path).name
        doc.metadata['file_type'] = ext.replace('.', '')
    
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
            
            for doc in documents:
                doc.metadata['source_file'] = pdf_file.name
                doc.metadata['file_type'] = 'pdf'
                
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
    
    if split_docs:
        print(f"\nExample chunk:")
        print(f"\t1. Content: {split_docs[0].page_content[:100]}")
        print(f"\t2. Metadata: {split_docs[0].metadata}")
    
    return split_docs

