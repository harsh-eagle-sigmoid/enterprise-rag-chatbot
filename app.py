import os
import streamlit as st

from loaders.pdf_loader import load_pdf
from embeddings.embedding_model import get_embedding_model
from vectorstore.faiss_store import create_vectorstore
from chains.qa_chain import create_qa_chain
from memory.chat_memory import get_memory
from utils.pdf_export import export_chat_to_pdf

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Enterprise RAG Chatbot (100% Free)",
    layout="wide"
)

st.title("📄 Enterprise RAG Chatbot (100% Free)")

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- ROLE SELECTION ----------------
role = st.selectbox(
    "Select your role",
    ["User", "Admin"],
    help="Admin gets detailed answers, User gets simplified answers"
)

# ---------------- FILE UPLOADER ----------------
uploaded_files = st.file_uploader(
    "Upload one or more PDFs",
    type="pdf",
    accept_multiple_files=True
)

# ---------------- SHOW CHAT HISTORY ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- CHAT INPUT ----------------
query = st.chat_input("Ask a question about your documents…")

# ---------------- MAIN LOGIC ----------------
if uploaded_files:
    os.makedirs("data", exist_ok=True)
    all_documents = []

    for uploaded_file in uploaded_files:
        file_path = os.path.join("data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        docs = load_pdf(file_path)
        all_documents.extend(docs)

    embeddings = get_embedding_model()
    vectorstore = create_vectorstore(all_documents, embeddings)
    memory = get_memory()
    qa_chain = create_qa_chain(vectorstore, memory, role, all_documents)

    if query:
        # ---- USER MESSAGE ----
        st.session_state.messages.append(
            {"role": "user", "content": query}
        )

        with st.chat_message("user"):
            st.markdown(query)

        # ---- CALL RAG CHAIN ----
        response = qa_chain.invoke({"question": query})

        # ---- ANTI-HALLUCINATION ----
        if not response.get("source_documents"):
            answer = "Information not found in the provided documents."
        else:
            answer = response["answer"]

        # ---- ASSISTANT MESSAGE ----
        with st.chat_message("assistant"):
            st.markdown(answer)

            if response.get("source_documents"):
                with st.expander("View source documents"):
                    for i, doc in enumerate(response["source_documents"], 1):
                        source = doc.metadata.get("source", "Unknown file")
                        page = doc.metadata.get("page", 0)

                        st.markdown(
                            f"**Source {i}: {source} (Page {page + 1})**"
                        )
                        st.write(doc.page_content[:500] + "...")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

else:
    st.info("👆 Upload one or more PDFs to start chatting.")

# ---------------- EXPORT AT END OF CHAT ----------------
if st.session_state.messages:
    st.divider()
    st.markdown("### 📤 Export Conversation")

    pdf_path = export_chat_to_pdf(
        st.session_state.messages,
        role
    )

    with open(pdf_path, "rb") as f:
        st.download_button(
            label="Download chat as PDF",
            data=f,
            file_name=os.path.basename(pdf_path),
            mime="application/pdf"
        )
