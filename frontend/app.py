import streamlit as st
import requests
from typing import Any

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="RAG Assistant",
    layout="centered"
)

API_URL = "http://localhost:8000/query"
UPLOAD_URL = "http://localhost:8000/upload"


# -----------------------------
# TITLE
# -----------------------------
st.title("RAG Assistant (Chatbot)")
st.caption("Ask questions from your uploaded Files")


# -----------------------------
# SIDEBAR - UPLOAD PDF
# -----------------------------
st.sidebar.header("Upload Files")

uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    if st.sidebar.button("Upload"):
        try:
            files = [
                ("files", (uploaded_file.name, uploaded_file.getvalue(), "application/pdf"))
            ]
            with st.sidebar:
                with st.spinner("Uploading and indexing the file..."):
                    response = requests.post(UPLOAD_URL, files=files)

            if response.status_code == 200:
                payload = response.json()
                st.sidebar.success(payload.get("message", "PDF uploaded & processed successfully"))
                if payload.get("files_skipped"):
                    st.sidebar.info(f"Skipped existing files: {', '.join(payload['files_skipped'])}")
                if payload.get("documents_in_collection") is not None:
                    st.sidebar.caption(
                        f"Chunks created: {payload.get('chunks_created', 0)} | "
                        f"Documents in DB: {payload.get('documents_in_collection')}"
                    )
            else:
                st.sidebar.error(f"Upload failed: {response.text}")

        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")



# -----------------------------
# SESSION STATE (chat history)
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


def normalize_answer(payload: Any) -> tuple[str, list]:
    if isinstance(payload, dict):
        answer = payload.get("answer", "No response")
        sources = payload.get("sources", [])
        return str(answer), sources if isinstance(sources, list) else []
    return str(payload), []


def render_sources(sources: list) -> None:
    if not sources:
        return

    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for source in sources:
            source_name = source.get("source", "Unknown source")
            source_page = source.get("page", "Unknown")
            score = source.get("score")

            label = f"{source_name} - p. {source_page}"
            if isinstance(score, (int, float)):
                label += f" - {score:.2f}"

            st.caption(label)

# -----------------------------
# DISPLAY CHAT HISTORY
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_sources(msg.get("sources", []))

# -----------------------------
# USER INPUT
# -----------------------------
user_input = st.chat_input("Ask a question about your documents...")

if user_input:
    # 1. Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Call FastAPI backend
    try:
        with st.chat_message("assistant"):
            with st.spinner("Searching your documents..."):
                response = requests.post(
                    API_URL,
                    json={"question": user_input}
                )

        if response.status_code == 200:
            answer, sources = normalize_answer(response.json())
        else:
            answer = f"Error: {response.status_code}"
            sources = []

    except Exception as e:
        answer = f"Backend error: {str(e)}"
        sources = []

    # 3. Show assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
    st.rerun()
