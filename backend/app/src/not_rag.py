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


def no_rag(query):
    
    prompt = f"""
You are a helpful an AI assistant.


Instructions:
- Follow the task type strictly
- Be clear and relevant
- Do not hallucinate facts
- Keep the answer concise unless explanation is required

---


User query:
"{query}"

Answer:
"""

    response = llm.invoke(
        prompt,
        temperature=0.4
    )

    return {
        'answer': response.content,
        'sources': []
    }