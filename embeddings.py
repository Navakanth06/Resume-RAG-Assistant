import streamlit as st
from sentence_transformers import SentenceTransformer


@st.cache_resource
def load_embedding_model():
    """
    Load the embedding model once and reuse it
    across Streamlit reruns.
    """

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


model = load_embedding_model()


def create_document_embeddings(chunks):
    """
    Create embeddings for all resume chunks.

    Batch encoding is significantly faster than
    encoding chunks individually.
    """

    embeddings = model.encode(
        chunks,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embeddings.tolist()


def create_question_embedding(question):
    """
    Create an embedding for the user's question.
    """

    embedding = model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embedding.tolist()