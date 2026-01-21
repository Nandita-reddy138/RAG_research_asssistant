# rag_core.py
import numpy as np
import faiss
import ollama
from pypdf import PdfReader

# Global objects
index = None
chunks = []


# 1️⃣ Load PDF
def load_pdf(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"

    return text


# 2️⃣ Chunk text
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    chunk_id = 1

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        chunks.append({
            "id": chunk_id,
            "text": chunk
        })

        start = end - overlap
        chunk_id += 1

    return chunks


# 3️⃣ Create embeddings
def embed_texts(texts):
    embeddings = []

    for t in texts:
        response = ollama.embeddings(
            model="nomic-embed-text",
            prompt=t
        )
        embeddings.append(response["embedding"])

    return np.array(embeddings).astype("float32")


# 4️⃣ Build FAISS index
def build_faiss_index(new_chunks):
    global index, chunks

    chunks = new_chunks
    texts = [c["text"] for c in chunks]

    embeddings = embed_texts(texts)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)


# 5️⃣ Embed query
def embed_query(query):
    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=query
    )
    return np.array(response["embedding"]).astype("float32").reshape(1, -1)


# 6️⃣ Search FAISS
def search_faiss(query_embedding, top_k=3):
    D, I = index.search(query_embedding, top_k)
    return I[0]


# 7️⃣ Retrieve chunks
def retrieve_chunks(indices):
    return [chunks[i] for i in indices]


# 8️⃣ Ask LLaMA
def ask_llama(context, question):
    prompt = f"""
You are a helpful assistant.
Answer ONLY using the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


# 9️⃣ RAG pipeline
def rag_pipeline(question):
    if index is None:
        return {
            "answer": "No document uploaded yet.",
            "sources": []
        }

    query_embedding = embed_query(question)
    indices = search_faiss(query_embedding)

    retrieved = retrieve_chunks(indices)

    context = ""
    sources = []

    for i, chunk in enumerate(retrieved, start=1):
        context += f"[Source {i}]\n{chunk['text']}\n\n"
        sources.append(f"Source {i} – Chunk {chunk['id']}")

    answer = ask_llama(context, question)

    return {
        "answer": answer,
        "sources": sources
    }
