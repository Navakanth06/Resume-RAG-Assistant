import chromadb

from embeddings import create_question_embedding


# ---------------------------------------------------
# ChromaDB
# ---------------------------------------------------

# In-memory client: a resume is only needed while its
# session is open, so nothing is written to disk and
# resumes are never left behind between runs.
client = chromadb.EphemeralClient()


def _collection_name(session_id):

    return f"resume_{session_id}"


def _get_collection(session_id):

    # One collection per browser session, so two people
    # using the app never overwrite each other's resume.
    return client.get_or_create_collection(
        name=_collection_name(session_id),
        metadata={"hnsw:space": "cosine"}
    )


# ---------------------------------------------------
# Store Embeddings
# ---------------------------------------------------

def store_embeddings(chunks, embeddings, session_id):

    delete_collection(session_id)

    collection = _get_collection(session_id)

    # Zero-padded ids sort in the same order as the resume
    # ("chunk_10" would otherwise sort before "chunk_2").
    ids = [
        f"chunk_{i:04d}"
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
    )


def delete_collection(session_id):

    try:

        client.delete_collection(
            name=_collection_name(session_id)
        )

    except Exception:

        # Nothing stored for this session yet
        pass


# ---------------------------------------------------
# Semantic Retrieval
# ---------------------------------------------------

# Questions asking for complete resume sections
SECTION_WORDS = [
    "skills",
    "skill",
    "experience",
    "work experience",
    "internship",
    "internships",
    "projects",
    "project",
    "certificates",
    "certificate",
    "certifications",
    "certification",
    "education",
    "degree",
    "cgpa"
]

# Questions that need the whole resume
BROAD_WORDS = [
    "summar",
    "overview",
    "interview",
    "strength",
    "weakness",
    "improve"
]


def retrieve_chunks(question, session_id):

    collection = _get_collection(session_id)

    count = collection.count()

    if count == 0:
        return ""

    question_lower = question.lower()

    needs_more_context = any(
        word in question_lower
        for word in SECTION_WORDS + BROAD_WORDS
    )

    # For section and broad questions, retrieve more information
    if needs_more_context:
        top_k = min(10, count)
    else:
        top_k = min(5, count)

    question_embedding = (
        create_question_embedding(question)
    )

    results = collection.query(
        query_embeddings=[
            question_embedding
        ],
        n_results=top_k
    )

    ids = results.get("ids", [[]])[0]

    documents = results.get(
        "documents",
        [[]]
    )[0]

    # Put the retrieved chunks back in resume order
    # instead of similarity order.
    ordered = sorted(
        zip(ids, documents),
        key=lambda pair: pair[0]
    )

    cleaned = []

    for _, document in ordered:

        if document and document.strip():

            document = document.strip()

            if document not in cleaned:
                cleaned.append(document)

    return "\n\n".join(cleaned)