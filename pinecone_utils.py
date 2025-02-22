import pinecone
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load API keys from .env
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "pdf-rag-index"

if index_name not in pc.list_indexes().names():
    pc.create_index(
      name=index_name, 
      dimension=1536, 
      metric='cosine',
      spec=ServerlessSpec(
        cloud='aws',
        region='us-east-1'
      )
    )
index = pc.Index(index_name)

def store_embedding(unique_id, embedding, metadata):
    """ Stores an embedding in Pinecone. """
    index.upsert([(unique_id, embedding, metadata)])

def search_pinecone(query_embedding, top_k=3):
    """ Searches Pinecone and returns top matching text chunks. """
    return index.query(vector=query_embedding, top_k=top_k, include_metadata=True)["matches"]
