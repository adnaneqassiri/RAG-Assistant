class RAGRetriever:
    """
    Handles query-based retrieval from the vector store
    """
    
    def __init__(self, vector_store, embeddings_manager):
        """
        Initialize the retriever
        
        Args:
            - vector_store: Vector store containing document embeddings
            - embeddings_manager: Manager for generating query embeddings
        """
        self.vector_store = vector_store
        self.embeddings_manager = embeddings_manager
        
    def retrieve(self, query, top_k, score_threshold):
        """
        Retrieve relevant documents for a query
        
        Args:
            - query: The search query
            - top_k: Number of top results to return
            - score_threshold: Minimum similarity score threshold
            
        Returns:
            - List of dictionaries containing retrieved documents and metadata
        """
        
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, score threshold: {score_threshold}")
        
        # Generate query embeddings
        query_embeddings = self.embeddings_manager.generate_embeddings([query])[0]
        
        # Search in Vector DB
        try:
            results = self.vector_store.collection.query(
                query_embeddings = [query_embeddings.tolist()],
                n_results = top_k
            )
            
            # Process results
            retrieved_docs = []
            if results["documents"] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]
                
                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    # Convert distance to similarity score (ChromaDB uses cosine distance) 
                    similarity_score = 1 - distance
                    
                    if distance <= score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })
                print(f"Retrieved {len(retrieved_docs)} documents after filtering")
            else:
                print("No document found")
            
            return retrieved_docs
        except Exception as e:
            print(f"Error During retrieval: {e}")
            return []
