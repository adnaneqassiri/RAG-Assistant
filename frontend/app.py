import streamlit as st
import requests

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
st.title("RAG Assistant (PDF Chatbot)")
st.caption("Ask questions from your uploaded PDFs")


# -----------------------------
# SIDEBAR - UPLOAD PDF
# -----------------------------
st.sidebar.header("Upload PDF")

uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    if st.sidebar.button("Upload"):
        try:
            files = [
                ("files", (uploaded_file.name, uploaded_file.getvalue(), "application/pdf"))
            ]
            response = requests.post(UPLOAD_URL, files=files)

            if response.status_code == 200:
                st.sidebar.success("PDF uploaded & processed successfully")
            else:
                st.sidebar.error(f"Upload failed: {response.text}")

        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")



# -----------------------------
# SESSION STATE (chat history)
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# DISPLAY CHAT HISTORY
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

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
        response = requests.post(
            API_URL,
            json={"question": user_input}
        )

        if response.status_code == 200:
            answer = response.json().get("answer", "No response")
        else:
            answer = f"Error: {response.status_code}"

    except Exception as e:
        answer = f"Backend error: {str(e)}"

    # 3. Show assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)