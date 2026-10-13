from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text):
    """
    Split resume text into overlapping chunks
    optimized for semantic retrieval.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_text(text)

    return [
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
    ]