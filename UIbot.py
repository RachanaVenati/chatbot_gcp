import os
import time
import re
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

collection = client.collections.get("json_array_1")  # json_array_1 #god1

# Token counter for logging
def count_tokens(text):
    text = "" if text is None else str(text)
    try:
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def get_recent_conversation():
    history = st.session_state.chat_history[-10:]
    return "\n".join([
        f"{'User' if role == 'user' else 'Assistant'}: {msg}"
        for role, msg in history if msg.strip()
    ])

def llama_completion(prompt):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

def verify_documents(question, context):
    context = truncate_context_to_fit_tokens(context)
    prompt = (
        f"Do these documents fully support answering the question below? Answer with Yes or No ONLY.\n\n"
        f"Documents:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    return "yes" in llama_completion(prompt).lower()

def get_missing_info_query(question, context):
    context = truncate_context_to_fit_tokens(context)
    prompt = (
        f"What is missing from these documents to fully answer the question?\n"
        f"Generate a new query that could help retrieve the missing information.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nImproved Query:"
    )
    return llama_completion(prompt)

def generate_response(question, context):
    context = truncate_context_to_fit_tokens(context)
    history = get_recent_conversation()
    prompt = (
        f"You are a helpful assistant. Use the following context to answer the user's question using only the history or the context provided to you, and answer precisely sentence.\n\n"
        f"Conversation so far:\n{history}\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    return llama_completion(prompt)

def fallback_general_response(question):
    history = get_recent_conversation()
    prompt = (
        f"You are an assistant that helps users based on general knowledge and prior conversation.\n"
        f"Never hallucinate. Respond with 'I don't know' if unsure.\n\n"
        f"Conversation so far:\n{history}\n\nUser's question: {question}\n\nAnswer:"
    )
    return llama_completion(prompt)

def truncate_context_to_fit_tokens(context, limit=2500):
    tokens = count_tokens(context)
    while tokens > limit:
        context = context[:len(context) - 100]
        tokens = count_tokens(context)
    return context

def log_retrievals(query, retrieved_docs, round_id, token_count):
    with open("retrieval_logs.txt", "a", encoding="utf-8") as f:
        f.write(f"\n=== Retrieval Round {round_id} ===\n")
        f.write(f"Query: {query}\n")
        f.write(f"Tokens: {token_count}\n")
        for i, text in enumerate(retrieved_docs, 1):
            snippet = text[:500].replace("\n", " ").replace("\r", " ")
            f.write(f"\n--- Document {i} ---\nContent Snippet: {snippet}...\n")
        f.write("\n")

def retrieve_context(query, limit=8):
    if not query or not query.strip():
        return []

    try:
        results = collection.query.near_text(
            query=query,
            limit=limit,
            # If available in your client version, this ensures the field is returned:
            # return_properties=["content"]
        )
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return []

    docs = []
    for obj in getattr(results, "objects", []) or []:
        props = getattr(obj, "properties", {}) or {}
        text = props.get("content") or props.get("text") or props.get("body") or props.get("chunk") or ""
        # Coerce to string safely
        if isinstance(text, list):
            text = " ".join([str(t) for t in text if t is not None])
        elif not isinstance(text, str):
            text = "" if text is None else str(text)

        if text.strip():
            docs.append(text)

    return docs

# Detect chit-chat or acknowledgments
ACKNOWLEDGMENT_PATTERNS = [
    r"\bthanks?\b",
    r"\bthank you\b",
    r"\bok(ay)?\b",
    r"\bthat'?s helpful\b",
    r"\bgreat\b",
    r"\bgood\b",
    r"\bappreciate\b"
]

def is_acknowledgment(text):
    text = text.lower()
    return any(re.search(pattern, text) for pattern in ACKNOWLEDGMENT_PATTERNS)

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

            if is_acknowledgment(user_input):
                # ✅ Use OpenAI for conversational response
                full_answer = fallback_general_response(user_input)
            else:
                query = user_input
                final_context = ""
                max_retries = 2
                for i in range(max_retries):
                    docs = retrieve_context(query)
                    if not docs:
                       full_answer= fallback_general_response(user_input)
                    else:
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
