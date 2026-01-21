# app.py
import streamlit as st
from rag_core import (
    load_pdf,
    chunk_text,
    build_faiss_index,
    rag_pipeline
)

st.set_page_config(
    page_title="📘 PDF RAG Assistant",
    layout="centered"
)

st.title("📘 PDF Research Assistant (RAG)")
st.write("Upload a PDF and ask questions from it.")

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
)

if uploaded_file:
    with st.spinner("Reading PDF..."):
        text = load_pdf(uploaded_file)

    with st.spinner("Chunking document..."):
        chunks = chunk_text(text)

    with st.spinner("Building knowledge index..."):
        build_faiss_index(chunks)

    st.success(f"Loaded {len(chunks)} chunks into knowledge base")

    st.divider()

    question = st.text_input("Ask a question from the document:")

    if st.button("Ask"):
        if question.strip() == "":
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                result = rag_pipeline(question)

            st.subheader("📌 Answer")
            st.write(result["answer"])

            st.subheader("📚 Sources")
            for src in result["sources"]:
                st.write("•", src)
