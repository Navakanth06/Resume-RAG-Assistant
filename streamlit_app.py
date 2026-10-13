import hashlib
import os
import tempfile
import time
import uuid

import streamlit as st

from pdf_processor import extract_text
from chunking import chunk_text
from embeddings import create_document_embeddings
from vector_store import (
    store_embeddings,
    retrieve_chunks,
    delete_collection
)
from rag import generate_answer


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Resume RAG Assistant",
    page_icon="📄",
    layout="centered"
)


# =====================================================
# HEADER
# =====================================================

st.title("📄 Resume RAG Assistant")

st.write(
    "Upload your resume and ask questions about "
    "your experience, skills, projects, education, "
    "or certifications."
)


# =====================================================
# SESSION STATE
# =====================================================

defaults = {
    "processed": False,
    "uploaded_file_id": None,
    # Changing the uploader key clears the uploaded file
    "uploader_key": 0,
    # Each browser session gets its own vector collection
    "session_id": uuid.uuid4().hex,
    # Full resume text, used for complete section answers
    "resume_text": ""
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# =====================================================
# UPLOAD RESUME
# =====================================================

uploaded_file = st.file_uploader(
    "Upload Resume (PDF)",
    type=["pdf"],
    key=f"resume_uploader_{st.session_state.uploader_key}"
)


# =====================================================
# PROCESS UPLOADED RESUME
# =====================================================

if uploaded_file is not None:

    # Identify the file by its content

    file_bytes = uploaded_file.getvalue()

    file_id = hashlib.md5(
        file_bytes
    ).hexdigest()

    # Detect a genuinely new resume

    if file_id != st.session_state.uploaded_file_id:

        st.session_state.processed = False
        st.session_state.resume_text = ""

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(
                file_bytes
            )

            pdf_path = temp_file.name

        try:

            with st.spinner(
                "⚙️ Processing your resume..."
            ):

                # -------------------------------------
                # Extract text
                # -------------------------------------

                text = extract_text(
                    pdf_path
                )

                if not text.strip():

                    raise ValueError(
                        "The uploaded PDF contains no "
                        "readable text. If it is a scanned "
                        "image, please upload a text-based PDF."
                    )

                # -------------------------------------
                # Chunk text
                # -------------------------------------

                chunks = chunk_text(
                    text
                )

                if not chunks:

                    raise ValueError(
                        "Unable to create resume chunks."
                    )

                # -------------------------------------
                # Create embeddings
                # -------------------------------------

                embeddings = (
                    create_document_embeddings(
                        chunks
                    )
                )

                # -------------------------------------
                # Store embeddings
                # -------------------------------------

                store_embeddings(
                    chunks,
                    embeddings,
                    st.session_state.session_id
                )

            # Remember the file only after it was processed
            # successfully, so a failed upload can be retried.

            st.session_state.resume_text = text
            st.session_state.uploaded_file_id = file_id
            st.session_state.processed = True

            st.success(
                "✅ Resume processed successfully!"
            )

        except Exception as e:

            st.session_state.processed = False
            st.session_state.uploaded_file_id = None

            st.error(
                f"Processing Error:\n\n{e}"
            )

        finally:

            if os.path.exists(pdf_path):
                os.remove(pdf_path)


# =====================================================
# READY STATUS
# =====================================================

if st.session_state.processed:

    st.success(
        "🟢 RAG Assistant Ready"
    )


# =====================================================
# QUESTION AREA
# =====================================================

if st.session_state.processed:

    question = st.text_input(
        "Ask a question about your resume",
        placeholder=(
            "Example: What projects have I completed?"
        )
    )

    ask = st.button(
        "🚀 Get Answer",
        use_container_width=True
    )

    reset = st.button(
        "🔄 Upload New Resume",
        use_container_width=True
    )

    # -----------------------------------------------
    # Reset
    # -----------------------------------------------

    if reset:

        delete_collection(
            st.session_state.session_id
        )

        st.session_state.processed = False
        st.session_state.uploaded_file_id = None
        st.session_state.resume_text = ""

        # A new uploader key clears the old file;
        # otherwise the same PDF would be re-processed
        # immediately after the rerun.
        st.session_state.uploader_key += 1

        st.rerun()

    # -----------------------------------------------
    # Ask Question
    # -----------------------------------------------

    if ask:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            try:

                start_time = time.perf_counter()

                with st.spinner(
                    "🤖 Finding the answer..."
                ):

                    # -----------------------------
                    # Retrieval
                    # -----------------------------

                    context = retrieve_chunks(
                        question,
                        st.session_state.session_id
                    )

                    # -----------------------------
                    # Generation
                    # -----------------------------

                    answer = generate_answer(
                        question,
                        context,
                        full_text=st.session_state.resume_text
                    )

                    total_time = (
                        time.perf_counter()
                        - start_time
                    )

                # ---------------------------------
                # Display answer
                # ---------------------------------

                st.subheader(
                    "🤖 Answer"
                )

                st.write(answer)

                st.caption(
                    f"⚡ Response time: "
                    f"{total_time:.2f}s"
                )

                # ---------------------------------
                # Debug context
                # ---------------------------------

                with st.expander(
                    "🔍 Retrieved Context"
                ):

                    st.write(context)

            except Exception as e:

                st.error(
                    f"Error: {e}"
                )


else:

    st.info(
        "📂 Upload a PDF resume to begin."
    )