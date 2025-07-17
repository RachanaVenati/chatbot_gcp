import os
import time
import requests
import streamlit as st
from dotenv import load_dotenv
import openai
import tiktoken
import weaviate
from weaviate.classes.query import MetadataQuery

# Load .env variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Connect to remote Weaviate instance
client = weaviate.connect_to_custom(
    http_host="34.159.21.160",   # Replace with your VM's external IP
    http_port=8080,
    http_secure=False,
    grpc_host="34.159.21.160",
    grpc_port=50051,
    grpc_secure=False
)

collection = client.collections.get("json_array_1")

# Token counter for logging
def count_tokens(text):
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(enc.encode(text))

# Get recent conversation
def get_recent_conversation():
    history = st.session_state.chat_history[-10:]
    return "\n".join([
        f"{'User' if role == 'user' else 'Assistant'}: {msg}"
        for role, msg in history if msg.strip()
    ])



def llama_completion(prompt):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # Or "gpt-4" if you have access
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

# Check if documents are sufficient
def verify_documents(question, context):
    context = truncate_context_to_fit_tokens(context)
    prompt = (
        f"Do these documents fully support answering the question below? Answer with Yes or No ONLY.\n\n"
        f"Documents:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    return "yes" in llama_completion(prompt).lower()

# Generate improved query
def get_missing_info_query(question, context):
    context = truncate_context_to_fit_tokens(context)
    prompt = (
        f"What is missing from these documents to fully answer the question?\n"
        f"Generate a new query that could help retrieve the missing information.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nImproved Query:"
    )
    return llama_completion(prompt)

# Final answer generation
def generate_response(question, context):
    context = truncate_context_to_fit_tokens(context)
    history = get_recent_conversation()
    prompt = (
        f"You are a helpful assistant. Use the following context to answer the user's question using only the history or the context provided to you, and answer precisely sentence.\n\n"
        f"Conversation so far:\n{history}\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    return llama_completion(prompt)

# Fallback response
def fallback_general_response(question):
    history = get_recent_conversation()
    prompt = (
        f"You are an assistant that helps users based on general knowledge and prior conversation.\n"
        f"Never hallucinate. Respond with 'I don't know' if unsure.\n\n"
        f"Conversation so far:\n{history}\n\nUser's question: {question}\n\nAnswer:"
    )
    return llama_completion(prompt)

# Truncate context to fit model token limits
def truncate_context_to_fit_tokens(context, limit=2500):
    tokens = count_tokens(context)
    while tokens > limit:
        context = context[:len(context) - 100]
        tokens = count_tokens(context)
    return context

# Log retrieval
def log_retrievals(query, retrieved_docs, round_id, token_count):
    with open("retrieval_logs.txt", "a", encoding="utf-8") as f:
        f.write(f"\n=== Retrieval Round {round_id} ===\n")
        f.write(f"Query: {query}\n")
        f.write(f"Tokens: {token_count}\n")
        for i, text in enumerate(retrieved_docs, 1):
            snippet = text[:500].replace("\n", " ").replace("\r", " ")
            f.write(f"\n--- Document {i} ---\nContent Snippet: {snippet}...\n")
        f.write("\n")

# Retrieve context from Weaviate
def retrieve_context(query, limit=8):
    results = collection.query.near_text(
        query=query,
        limit=limit
    )
    return [obj.properties["text"] for obj in results.objects]

# Streamlit UI
st.set_page_config(page_title="RAG Chatbot (OpenAI + Weaviate)", layout="centered")
st.title("\U0001F9E0 RAG Chatbot using OpenAI + Weaviate")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for role, message in st.session_state.chat_history:
    avatar = "\U0001F9D1‍\U0001F4BB" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(f"""
            <div style='background-color:#1e1e20;padding:1rem;border-radius:0.5rem;border:1px solid #333;'>
                {message}
            </div>
        """, unsafe_allow_html=True)

# Handle user input
user_input = st.chat_input("Ask your question...")
if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("user", avatar="\U0001F9D1‍\U0001F4BB"):
        st.markdown(
            f"""
            <div style='background-color:#1e1e20;padding:1rem;border-radius:0.5rem;border:1px solid #333;'>
                {user_input}
            </div>
            """, unsafe_allow_html=True
        )

    st.session_state.chat_history.append(("bot", ""))
    with st.chat_message("assistant", avatar="🤖"):
        msg_placeholder = st.empty()
        with st.spinner("Retrieving context and generating response..."):
            query = user_input
            final_context = ""
            max_retries = 2
            for i in range(max_retries):
                docs = retrieve_context(query)
                if not docs:
                    break
                context = "\n".join(docs)
                log_retrievals(query, docs, i + 1, token_count=count_tokens(context))
                if verify_documents(user_input, context):
                    final_context = context
                    break
                else:
                    query = get_missing_info_query(user_input, context)
                    time.sleep(1.2)

            if final_context:
                full_answer = generate_response(user_input, final_context)
            else:
                full_answer = fallback_general_response(user_input)

        typed_text = ""
        for char in full_answer:
            typed_text += char
            msg_placeholder.markdown(
                f"""
                <div style='background-color:#1e1e20;padding:1rem;border-radius:0.5rem;border:1px solid #333;'>
                    {typed_text}▌
                </div>
                """, unsafe_allow_html=True
            )
            time.sleep(0.015)

        msg_placeholder.markdown(
            f"""
            <div style='background-color:#1e1e20;padding:1rem;border-radius:0.5rem;border:1px solid #333;'>
                {typed_text}
            </div>
        """, unsafe_allow_html=True)

        st.session_state.chat_history[-1] = ("bot", full_answer)
