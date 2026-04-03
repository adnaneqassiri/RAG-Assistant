from sentence_transformers import SentenceTransformer



class EmbManagaer():
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the embeddings manager
        
        Args:
            - model_name: HuggingFace model name for sentence embeddings    
        """
        
        self.model_name = model_name
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """
        Load The SentenceTransformer model
        """
        try:
            print(f'Loading the model: {self.model_name}')
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded sccessfully, emb dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise 
        
        
    def generate_embeddings(self, texts):
        """
        Generate embeddings ofr a list of texts
        
        Args:
            - texts: List of string texts

        Returns:
            - numpy array of embeddings with shape (len(texts), embeddings_dim)
        """
        
        if not self.model:
            raise ValueError('Model not Loaded')
        
        print(f"Generating embeddings for {len(texts)} texts ...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        
        return embeddings


