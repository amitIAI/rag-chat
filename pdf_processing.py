import fitz
from nltk.tokenize import sent_tokenize

def get_pdf_text(pdf_path):
    """ Extracts text from a PDF and returns a list of sentences. """
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return sent_tokenize(text)

def chunk_text(sentences, chunk_size=1000):
    """ Breaks text into smaller chunks while keeping sentences intact. """
    chunks, current_chunk, current_length = [], [], 0
    for sentence in sentences:
        if current_length + len(sentence) > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_length = [], 0
        current_chunk.append(sentence)
        current_length += len(sentence)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks
