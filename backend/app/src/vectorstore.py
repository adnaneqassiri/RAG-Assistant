import os
import chromadb
import hashlib
from pathlib import Path

def generate_doc_id(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

class VectorStore:
    """
    Manages the vector store in a ChromeDB vector db
    """
    def __init__(self, collection_name, persist_directory):
        """
        Init the vector store

        Args: 
            - collection_name: Name of Chroma collection
            - persist_directory; Directory in which we persist the db
        """
    
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()
    
    def _initialize_store(self):
        """
        Initialize ChromaDB Client and Collection
        """
        try:
            resolved_directory = str(Path(self.persist_directory).resolve())
            os.makedirs(resolved_directory, exist_ok=True)
            self.persist_directory = resolved_directory
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "PDF document embeddings for RAG"
                }    
            )
            
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Persist directory: {self.persist_directory}")
            print(f"Existing documents in collection: {self.collection.count()}")
            
        except Exception as e:
            print(f"Error intializing vector store: {e}")
            raise
    
    def add_document(self, documents, embeddings):
        """
        Add documents and thier embeddings to the vector store
        
        Args:
            - documents: List of langchain document
            - embeddings: Correspoding embeddings for those documents
        """
        
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        print(f"Adding {len(documents)} documents to vector store...")
        
        # Prepare data for ChromaDB
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []
        
        
        for i, (doc, embeddings) in enumerate(zip(documents, embeddings)):
            metadata = dict(doc.metadata)

            file_hash = metadata.get("file_hash", "")
            chunk_index = metadata.get("chunk_index", i)
            doc_id = f"{file_hash}:{chunk_index}" if file_hash else generate_doc_id(doc.page_content)
            ids.append(doc_id)
            
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)
            
            # Document content
            documents_text.append(doc.page_content)
            
            # Embeddings 
            embeddings_list.append(embeddings.tolist())
            
        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            
            print(f"Successfully added {len(documents)} document to the DB")
            print(f"Total document in the collection: {self.collection.count()}")
            
        except Exception as e:
            print(f"Error adding documents: {e}")
            raise

    def document_exists(self, file_hash=None, source_file=None):
        where = {}
        if file_hash:
            where["file_hash"] = file_hash
        elif source_file:
            where["source_file"] = source_file
        else:
            return False

        results = self.collection.get(where=where, include=["metadatas"])
        return bool(results.get("ids"))

    def get_documents_by_metadata(self, where=None, limit=None):
        results = self.collection.get(
            where=where or None,
            include=["metadatas", "documents", "embeddings"]
        )

        ids = results.get("ids", [])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        embeddings = results.get("embeddings", [])

        records = []
        for doc_id, document, metadata, embedding in zip(ids, documents, metadatas, embeddings):
            records.append({
                "id": doc_id,
                "content": document,
                "metadata": metadata or {},
                "embedding": embedding,
            })

        if limit is not None:
            return records[:limit]
        return records
