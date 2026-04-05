import json
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

# Initialiser UNE FOIS
llm = ChatGroq(
    api_key=groq_api_key,
    model="llama-3.3-70b-versatile",
    max_tokens=1024
)

def simple_router(query):
    prompt = f"""
You are a query router for a Retrieval-Augmented Generation (RAG) system.
Return ONLY JSON. Do not explain anything. Do not add text.


Your task is to classify the user query and decide:
1. Whether document retrieval is needed
2. The type of task
3. The appropriate generation temperature

---

TASK TYPES:
- factual_qa → precise question, requires accurate answer
- explanation → explain a concept clearly
- summarization → summarize content
- rewrite → rephrase or improve text
- creative_generation → open-ended or creative writing

---

RULES:
- If the query refers to documents, files, PDFs, or says "according to the document", retrieval_needed = true
- If the query is general knowledge, retrieval_needed = false
- factual_qa → temperature = 0.2
- explanation → temperature = 0.3
- summarization → temperature = 0.2
- rewrite → temperature = 0.5
- creative_generation → temperature = 0.7

---

Examples:

Query: "What is machine learning?"
Output:
{{
  "retrieval_needed": false,
  "task_type": "factual_qa",
  "temperature": 0.2
}}

Query: "Summarize this document"
Output:
{{
  "retrieval_needed": true,
  "task_type": "summarization",
  "temperature": 0.2
}}

Query: "Write a creative introduction about AI"
Output:
{{
  "retrieval_needed": false,
  "task_type": "creative_generation",
  "temperature": 0.7
}}


---

Return ONLY a valid JSON object with this format:

{{
  "retrieval_needed": true/false,
  "task_type": "...",
  "temperature": number
}}

---

User query:
"{query}"
"""
    
    response = llm.invoke(prompt)
    
    try:
        return json.loads(response.content)
    except:
        # fallback simple
        return {
            "retrieval_needed": True,
            "task_type": "factual_qa",
            "temperature": 0.2
        }