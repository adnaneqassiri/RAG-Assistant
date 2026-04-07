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


def _format_history(history, max_turns=5):
    if not history:
        return "No previous conversation."

    formatted_turns = []
    for item in history[-max_turns:]:
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        formatted_turns.append(f"User: {question}\nAssistant: {answer}")

    return "\n\n".join(formatted_turns)


def no_rag(query, history=None):
    conversation_history = _format_history(history or [])
    prompt = f"""
You are a helpful an AI assistant.

Instructions:
- Follow the task type strictly
- Be clear and relevant
- Do not hallucinate facts
- Keep the answer concise unless explanation is required
- Use the conversation history when the user refers to earlier messages

Conversation history:
{conversation_history}

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
