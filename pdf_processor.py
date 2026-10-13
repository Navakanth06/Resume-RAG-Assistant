from pypdf import PdfReader


def extract_text(pdf_path):
    """
    Extract readable text from all pages of a PDF.
    """

    reader = PdfReader(pdf_path)

    pages = []

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            pages.append(page_text.strip())

    return "\n\n".join(pages)