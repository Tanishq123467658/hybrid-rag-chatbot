import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from rag_agent import agent, DocumentLoader

# Model
model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

st.set_page_config(page_title="Hybrid RAG Chatbot", layout="wide")

st.title("📄 Hybrid RAG Chatbot")

# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# 📂 SIDEBAR (UPLOAD)
# =========================
with st.sidebar:
    st.header("📂 Upload Document")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type="pdf"
    )

    if uploaded_file is not None:

        if "last_file" not in st.session_state or st.session_state.last_file != uploaded_file.name:

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                path = tmp.name

            result = DocumentLoader.invoke({"file_path": path})

            st.session_state.last_file = uploaded_file.name

            # Add to chat (important UX)
            st.session_state.messages.append(
                HumanMessage(content=f"📎 Uploaded: {uploaded_file.name}")
            )
            st.session_state.messages.append(
                AIMessage(content="Document processed successfully ✅")
            )

            st.success("Document loaded!")
            st.rerun()

    # Optional: clear chat
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.success("Chat cleared!")
        st.rerun()

# =========================
# 💬 CHAT DISPLAY
# =========================
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# =========================
# 💬 CHAT INPUT (BOTTOM)
# =========================
user_input = st.chat_input("Ask something...")

# =========================
# 🧠 CHAT LOGIC
# =========================
if user_input:

    st.session_state.messages.append(HumanMessage(content=user_input))

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            response = agent(
                model=model,
                messages=st.session_state.messages
            )

            if not response or response.strip() == "":
                response = "I couldn't generate a response."

            st.write(response)

    st.session_state.messages.append(AIMessage(content=response))