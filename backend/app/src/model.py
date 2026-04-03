import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
import time


class GroqLLM:
    def __init__(self, model_name="llama-3.3-70b-versatile", api_key = None):
        """
        Initialize Groq LLM
        
        Args:
            - model_name: Groq model name 
            - api_key: Groq api key
        """
        
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError("Groq API key is required.")
        
        self.llm = ChatGroq(
            model = self.model_name,
            api_key = self.api_key,
            temperature= 0.1,
            max_tokens=1024
        )
        
        print(f"Initialized GROQ LLM with model: {self.model_name}")
        
    
    def generate_response(self, query, context, max_length=500):
        """
        Generate Response using retrieved context
        
        Args:
            - query: User question
            - context: Retrieved document context
            - max_length: Maximum response length
            
        Returns:
            - Generated response string
        """
        
        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are a helpful AI assistant. Use the following context, to answer the question accurately and concisely.

            Context: {context}
            
            Question: {question}
            
            Answer: Provide a clear and informative answer based on the context above. if the context doesnt contain enough information to answer the question, please say so.
            """
        )
        
        # format the prompt
        formatted_prompt = prompt_template.format(context=context, question=query)
        
        try:
            # Generate the response
            messages = [HumanMessage(content=formatted_prompt)]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error generating the response: {str(e)}"
    
    def summarize(self, query):
        """
        Simple response generation without complex prompting
        Args:
            - query: User question
            
        Returns:
            - Summary      
        """

        try:
            messages = [HumanMessage(content=query)]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"



class AdvancedRAGPipline:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.history = []
    
    def query(self, question, top_k=5, min_score=2.0, stream=False, summarize=False):
        # Retriever relevant documents
        results = self.retriever.retrieve(question, top_k=top_k, score_threshold=min_score)
        if not results:
            answer = "No relevant context found"
            sources = []
            context = ''
        else:
            context = '\n\n'.join([doc['content']  for doc in results])
            sources = [{
                'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unkown')),
                'page': doc['metadata'].get('page', 'unkown'),
                'score': doc['similarity_score'],
                'preview': doc['content']
            } for doc in results]
            
            # Streaming answer simulation
            prompt = f"""
            Use the following context to answer the question concisely.
            Context: {context}
            Question: {question}
            Answer:
            """
            
            if stream:
                print("Streaming answer:")
                for i in range(0, len(prompt), 80):
                    print(prompt[i:i+80], end='', flush=True)
                    time.sleep(0.05)
                print()
            response = self.llm.generate_response(question, context, max_length=500)
            answer = response
            
        # Add citations to answer
        citations = [f"[{i+1}] {src['source']} (page {src['page']})" for i, src in enumerate(sources)]
        answer_with_citations = answer + "\n\nCitation:\n" + '\n'.join(citations) if citations else answer
        
        # Optionally summarize the answer
        summary = None
        if summarize and answer:
            summary_prompt = f"Summarize the following answer in 2 sentences: \n{answer}"
            summary_resp = self.llm.summarize(summary_prompt)
            summary = summary_resp
        
        # Store query history
        self.history.append({
            'question': question,
            'answer': answer,
            'sources': sources,
            'summary': summary
        })
        
        return {
            'question': question,
            'answer': answer_with_citations,
            'sources': sources,
            'summary': summary,
            'history': self.history
        }
        
        
