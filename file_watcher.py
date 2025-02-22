import os
import time
import hashlib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pdf_processing import get_pdf_text, chunk_text
from embeddings import generate_embedding
from pinecone_utils import store_embedding

PDF_FOLDER = "pdfs"

class PDFMonitor(FileSystemEventHandler):
    """ Watches the PDF folder and updates Pinecone when files change. """

    def on_created(self, event):
        if event.src_path.endswith(".pdf"):
            time.sleep(2)  # 🔹 Wait 2s to ensure file is fully copied
            process_pdf(event.src_path)

    def on_deleted(self, event):
        if event.src_path.endswith(".pdf"):
            print(f"📁 PDF deleted: {os.path.basename(event.src_path)}")

def process_pdf(pdf_path):
    """ Extracts text, chunks, generates embeddings, and stores in Pinecone. """
    print(f"🔍 Processing {pdf_path}...")

    sentences = get_pdf_text(pdf_path)
    chunks = chunk_text(sentences)

    for chunk in chunks:
        embedding = generate_embedding(chunk)
        unique_id = hashlib.md5(chunk.encode()).hexdigest()
        metadata = {"source": os.path.basename(pdf_path), "text": chunk}
        
        store_embedding(unique_id, embedding, metadata)

    print(f"✅ Stored {len(chunks)} chunks from {pdf_path} in Pinecone.")

def process_existing_pdfs():
    print("✅ Existing PDFs processed.")
    """ Process all existing PDFs in the 'pdfs' folder at startup. """
"""
    print("📂 Checking for existing PDFs...")
    
    for file in os.listdir(PDF_FOLDER):
        if file.endswith(".pdf"):
            file_path = os.path.join(PDF_FOLDER, file)
            process_pdf(file_path)

    print("✅ Existing PDFs processed.")
"""

def start_monitoring():
    """ Start monitoring the folder for new, modified, or deleted PDFs. """
    observer = Observer()
    event_handler = PDFMonitor()
    observer.schedule(event_handler, PDF_FOLDER, recursive=True)
    observer.start()
    
    print("🚀 Monitoring 'pdfs' folder for changes...")

    try:
        while True:
            time.sleep(1)  # Keep script running
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# 🔹 Step 1: Process all existing PDFs on startup
process_existing_pdfs()

# 🔹 Step 2: Start monitoring for new PDFs
start_monitoring()