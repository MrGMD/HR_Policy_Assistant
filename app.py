import os
import tempfile

import faiss
import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📘",
    layout="wide",
)

st.title("📘 HR Policy Assistant")
st.caption("Upload an HR policy PDF and ask questions using retrieval-augmented generation (RAG).")

# -----------------------------
# Configuration
# -----------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 5


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def extract_pdf_text(uploaded_file):
    """Extract text from every page and keep page numbers for citations."""
    pdf_bytes = uploaded_file.getvalue()
    pages = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append(
                    {
                        "page": page_number,
                        "text": text,
                    }
                )

    return pages


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Create overlapping word-based chunks."""
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = max(end - overlap, start + 1)

    return chunks


def build_chunks(pages):
    """Create chunks while preserving the source page number."""
    chunks = []

    for page_data in pages:
        page_chunks = chunk_text(page_data["text"])

        for chunk in page_chunks:
            chunks.append(
                {
                    "text": chunk,
                    "page": page_data["page"],
                }
            )

    return chunks


def build_faiss_index(chunks, model):
    texts = [item["text"] for item in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


def retrieve_chunks(question, index, chunks, model, top_k=TOP_K):
    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    question_embedding = np.asarray(question_embedding, dtype="float32")

    k = min(top_k, len(chunks))
    scores, indices = index.search(question_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        results.append(
            {
                "text": chunks[idx]["text"],
                "page": chunks[idx]["page"],
                "score": float(score),
            }
        )

    return results


def create_prompt(question, retrieved_chunks):
    context_parts = []

    for i, item in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Source {i} | Page {item['page']}]\n{item['text']}"
        )

    context = "\n\n".join(context_parts)

    return f"""
You are an HR Policy Assistant.

Answer the user's question using ONLY the HR policy context provided below.

Rules:
1. Do not invent or assume policy information.
2. If the answer is not supported by the provided context, clearly say:
   "I could not find this information in the uploaded HR policy."
3. Give a concise, professional answer.
4. When useful, mention the relevant policy page number.
5. If the policy contains conditions, exceptions, limits, or approval requirements, include them.
6. Do not provide legal advice. If the question requires legal interpretation beyond the policy, recommend consulting HR or qualified legal counsel.

HR POLICY CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
""".strip()


def ask_groq(question, retrieved_chunks, client):
    prompt = create_prompt(question, retrieved_chunks)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer HR policy questions accurately and only from "
                    "the supplied policy context."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.1,
        max_tokens=700,
    )

    return response.choices[0].message.content


# -----------------------------
# Session state
# -----------------------------
if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "file_name" not in st.session_state:
    st.session_state.file_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Settings")

    api_key = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="For deployment, store GROQ_API_KEY in Streamlit Secrets.",
    )

    st.markdown("---")
    st.write("**RAG Pipeline**")
    st.write("1. Upload PDF")
    st.write("2. Extract text with PyMuPDF")
    st.write("3. Create overlapping chunks")
    st.write("4. Generate Sentence Transformer embeddings")
    st.write("5. Store vectors in FAISS")
    st.write("6. Retrieve relevant chunks")
    st.write("7. Generate answer with Groq")

    if st.session_state.file_name:
        st.markdown("---")
        st.success(f"Loaded: {st.session_state.file_name}")
        st.write(f"Chunks: {len(st.session_state.chunks)}")

        if st.button("Clear document", use_container_width=True):
            st.session_state.index = None
            st.session_state.chunks = []
            st.session_state.file_name = None
            st.session_state.messages = []
            st.rerun()


# -----------------------------
# PDF Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload HR Policy PDF",
    type=["pdf"],
    help="Upload a text-based HR policy PDF.",
)

if uploaded_file is not None:
    if uploaded_file.name != st.session_state.file_name:
        with st.spinner("Processing HR policy PDF..."):
            pages = extract_pdf_text(uploaded_file)

            if not pages:
                st.error(
                    "No readable text was found in this PDF. "
                    "If it is a scanned PDF, OCR is required before using this app."
                )
            else:
                chunks = build_chunks(pages)

                if not chunks:
                    st.error("No text chunks could be created from the PDF.")
                else:
                    model = load_embedding_model()
                    index = build_faiss_index(chunks, model)

                    st.session_state.index = index
                    st.session_state.chunks = chunks
                    st.session_state.file_name = uploaded_file.name
                    st.session_state.messages = []

                    st.success(
                        f"Policy loaded successfully: {len(pages)} pages, "
                        f"{len(chunks)} searchable chunks."
                    )


# -----------------------------
# Main Chat
# -----------------------------
st.markdown("### Ask about the policy")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input(
    "Example: How many annual leave days are employees entitled to?"
)

if question:
    if st.session_state.index is None:
        st.warning("Please upload an HR policy PDF first.")
        st.stop()

    if not api_key:
        st.error(
            "Please provide a Groq API key in the sidebar or configure "
            "GROQ_API_KEY in Streamlit Secrets."
        )
        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    model = load_embedding_model()

    with st.chat_message("assistant"):
        with st.spinner("Searching the policy..."):
            retrieved_chunks = retrieve_chunks(
                question,
                st.session_state.index,
                st.session_state.chunks,
                model,
                TOP_K,
            )

        try:
            client = Groq(api_key=api_key)

            with st.spinner("Generating answer..."):
                answer = ask_groq(question, retrieved_chunks, client)

            st.markdown(answer)

            st.markdown("#### Sources")

            for i, item in enumerate(retrieved_chunks, start=1):
                with st.expander(
                    f"Source {i} — Page {item['page']} — Similarity {item['score']:.3f}"
                ):
                    st.write(item["text"])

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

        except Exception as exc:
            st.error(f"Unable to generate the answer: {exc}")
