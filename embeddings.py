import openai
import os
import numpy as np
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load API keys from .env
load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_EMBEDDINGS_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME")
AZURE_EMBEDDING_VERSION = os.getenv("AZURE_OPENAI_EMBEDDINGS_VERSION")

def generate_embedding(text, model="text-embedding-ada-002"):
    """ Generate embeddings using Azure OpenAI. """
    client = AzureOpenAI(
    	api_key = AZURE_OPENAI_API_KEY,
    	api_version = AZURE_EMBEDDING_VERSION,
    	azure_endpoint = AZURE_OPENAI_ENDPOINT
    )
    
    response = client.embeddings.create( input=text, model=model )
    return response.data[0].embedding
