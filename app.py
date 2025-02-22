import streamlit as st
import openai
import os
from dotenv import load_dotenv
from embeddings import generate_embedding
from pinecone_utils import search_pinecone
from openai import AzureOpenAI

# Load API keys
load_dotenv()
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_CHAT_DEPLOYMENT_NAME")
AZURE_CHAT_VERSION = os.getenv("OPENAI_API_VERSION")

# Create OpenAI client
client = AzureOpenAI( api_key=AZURE_OPENAI_API_KEY )

st.title("📚 Welcome to Heartfulness GPT")

# 🔹 Initialize chat history if not present
if "messages" not in st.session_state:
    st.session_state.messages = []

# 🔹 Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 🔹 Get user input
query = st.chat_input("Hello, my name is Ira. What do you want to know about heartfulness meditation today?")

if query:
    # 🔹 Immediately add user input to chat history BEFORE processing
    st.session_state.messages.append({"role": "user", "content": query})

    # 🔹 Display user input in chat box
    with st.chat_message("user"):
        st.write(query)

    # Generate embedding for query
    query_embedding = generate_embedding(query)

    # Retrieve top 3 relevant chunks from Pinecone
    results = search_pinecone(query_embedding)

    # 🔹 Construct prompt with retrieved context
    context = "\n\n".join([match["metadata"]["text"] for match in results])
    prompt = f"Context:\n{context}\n\nAnswer the following based on the context: {query}"

    # Get response from Azure OpenAI
    response = client.chat.completions.create(
        model=AZURE_CHAT_DEPLOYMENT_NAME,
        messages=[{"role": "system", "content": "You are a helpful AI assistant."},
                  {"role": "user", "content": prompt}]
    )

    ai_response = response.choices[0].message.content

    # 🔹 Immediately add AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": ai_response})

    # 🔹 Display AI response in chat box
    with st.chat_message("assistant"):
        st.write(ai_response)