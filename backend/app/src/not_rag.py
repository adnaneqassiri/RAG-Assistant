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


def no_rag(query, task_type, temperature):
    
    prompt = f"""
You are an AI assistant.

Your task is to answer the user query based on the specified task type.

---

Task types:
- factual_qa → give a precise and correct answer
- explanation → explain clearly in simple terms
- summarization → summarize concisely
- rewrite → improve or rephrase the text
- creative_generation → generate creative content

---

Instructions:
- Follow the task type strictly
- Be clear and relevant
- Do not hallucinate facts
- Keep the answer concise unless explanation is required

---

Task type: {task_type}

User query:
"{query}"

Answer:
"""

    response = llm.invoke(
        prompt,
        temperature=temperature
    )

    return {
        'answer': response.content,
        'sources': []
    }