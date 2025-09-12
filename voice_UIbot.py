# app.py
import os
import time
import re
import hashlib
import streamlit as st
from dotenv import load_dotenv
import openai
import tiktoken
import weaviate

# 🔊 Voice helpers (your existing module)
from audio_recorder_streamlit import audio_recorder
from voice_assistant import (
    transcribe_audio_bytes,
    synthesize_speech_mp3,
    SimpleRateLimiter,
)

# =========================
# App setup & configuration
# =========================
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(
    page_title="RAG Chatbot (OpenAI + Weaviate) + Voice",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------- UI Chrome ---------
st.markdown("""
<style>
    /* Root layout */
    .main { display: flex; flex-direction: column; height: 100vh; }
    /* Scrollable chat area, leave room for the fixed input bar */
    .chat-container {
        flex: 1;
        overflow-y: auto;
        padding: 1rem;
        padding-bottom: 110px; /* keep last bubble above bar */
    }
    /* Fixed bottom input bar */
    .input-container {
        position: fixed;
        bottom: 10px; /* exact gap from bottom */
        left: 0;
        right: 0;
        background: #0e1117;
        padding: 10px 12px;
        border-top: 1px solid #2b303b;
        z-index: 1000;
        pointer-events: none; /* only inner row receives events */
    }
    .input-inner {
        max-width: 900px;
        margin: 0 auto;
        display: flex;
        gap: 10px;
        align-items: center;
        pointer-events: auto;
        background: #1f2023;
        border: 1px solid #3a3b3e;
        border-radius: 9999px;
        padding: 8px 10px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.35);
    }
    .text-input { flex: 1; }
    .text-input .stTextInput > div > div > input {
        background:#2a2b2e; border:1px solid #3a3b3e; color:#fff;
        border-radius: 9999px; height:44px; padding-left: 14px;
    }
    .send-col .stButton > button { height:44px; border-radius:9999px; padding:0 18px; }
    /* Make audio recorder minimal & centered */
    .mic-col .stVerticalBlock { display:flex; justify-content:center; align-items:center; height:44px; }
    .audio-recorder { margin:0 !important; background:transparent !important; box-shadow:none !important; border:none !important; }
    .audio-recorder svg { width:24px !important; height:24px !important; }
    /* Chat bubbles spacing */
    .stChatMessage { margin-bottom: 12px !important; }
</style>
""", unsafe_allow_html=True)

st.title("Weaviate + 🎙️ Voice")

# Sidebar: audio output toggle
with st.sidebar:
    st.markdown("### 🎧 Audio Output")
    speak_replies = st.toggle("Speak assistant replies", value=False)
    st.caption("Turn on to hear the assistant’s answers.")

# -------------------------
# Session state & constants
# -------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_user_input" not in st.session_state:
    st.session_state.pending_user_input = None
if "clear_text_next_run" not in st.session_state:
    st.session_state.clear_text_next_run = False  # safe clear pattern
if "last_audio_digest" not in st.session_state:
    st.session_state.last_audio_digest = None
if "just_sent" not in st.session_state:
    st.session_state.just_sent = False          # true only right after Send
if "scroll_to_bottom" not in st.session_state:
    st.session_state.scroll_to_bottom = False

TEXT_KEY = "composer_text_ui_v1"
MIC_KEY  = "composer_mic_ui_v1"
MIN_AUDIO_BYTES = 2000  # ignore tiny/empty mic captures

# =========================
# Weaviate connection
# =========================
WEAVIATE_HTTP_HOST = os.getenv("WEAVIATE_HTTP_HOST", "34.159.21.160")
WEAVIATE_HTTP_PORT = int(os.getenv("WEAVIATE_HTTP_PORT", "8080"))
WEAVIATE_HTTP_SECURE = os.getenv("WEAVIATE_HTTP_SECURE", "false").lower() == "true"
WEAVIATE_GRPC_HOST = os.getenv("WEAVIATE_GRPC_HOST", WEAVIATE_HTTP_HOST)
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
WEAVIATE_GRPC_SECURE = os.getenv("WEAVIATE_GRPC_SECURE", "false").lower() == "true"
WEAVIATE_COLLECTION = os.getenv("WEAVIATE_COLLECTION", "json_array_1")

client = weaviate.connect_to_custom(
    http_host=WEAVIATE_HTTP_HOST,
    http_port=WEAVIATE_HTTP_PORT,
    http_secure=WEAVIATE_HTTP_SECURE,
    grpc_host=WEAVIATE_GRPC_HOST,
    grpc_port=WEAVIATE_GRPC_PORT,
    grpc_secure=WEAVIATE_GRPC_SECURE
)
collection = client.collections.get(WEAVIATE_COLLECTION)

# =========================
# Core helpers (same behavior as before)
# =========================
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
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()

def truncate_context_to_fit_tokens(context, limit=2500):
    tokens = count_tokens(context)
    while tokens > limit:
        context = context[:len(context) - 100]
        tokens = count_tokens(context)
    return context

def verify_documents(question, context):
    context = truncate_context_to_fit_tokens(context)
    prompt = (
        "Do these documents fully support answering the question below? "
        "Answer with Yes or No ONLY.\n\n"
        f"Documents:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    return "yes" in llama_completion(prompt).lower()

def get_missing_info_query(question, context):
    context = truncate_context_to_fit_tokens(context)
    prompt = (
        "What is missing from these documents to fully answer the question?\n"
        "Generate a new query that could help retrieve the missing information.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nImproved Query:"
    )
    return llama_completion(prompt)

def generate_response(question, context):
    context = truncate_context_to_fit_tokens(context)
    history = get_recent_conversation()
    prompt = (
        "You are a helpful assistant. Use the following context to answer the "
        "user's question using only the history or the context provided to you, "
        "and answer precisely sentence.\n\n"
        f"Conversation so far:\n{history}\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    return llama_completion(prompt)

def fallback_general_response(question):
    history = get_recent_conversation()
    prompt = (
        "You are an assistant that helps users based on general knowledge and prior conversation.\n"
        "Never hallucinate. Respond with 'I don't know' if unsure.\n\n"
        f"Conversation so far:\n{history}\n\nUser's question: {question}\n\nAnswer:"
    )
    return llama_completion(prompt)

def retrieve_context(query, limit=8):
    if not query or not query.strip():
        return []
    try:
        results = collection.query.near_text(query=query, limit=limit)
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return []
    docs = []
    for obj in getattr(results, "objects", []) or []:
        props = getattr(obj, "properties", {}) or {}
        text = props.get("content") or props.get("text") or props.get("body") or props.get("chunk") or ""
        if isinstance(text, list):
            text = " ".join([str(t) for t in text if t is not None])
        elif not isinstance(text, str):
            text = "" if text is None else str(text)
        if text.strip():
            docs.append(text)
    return docs

ACK_PATTERNS = [
    r"\bthanks?\b", r"\bthank you\b", r"\bok(ay)?\b",
    r"\bthat'?s helpful\b", r"\bgreat\b", r"\bgood\b", r"\bappreciate\b"
]
def is_ack(text):
    text = text.lower()
    return any(re.search(p, text) for p in ACK_PATTERNS)

# =========================
# Chat history (scrollable area)
# =========================
with st.container():
    st.markdown('<div class="chat-container" id="chat-container">', unsafe_allow_html=True)
    for role, message in st.session_state.chat_history:
        avatar = "🧑‍💻" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.markdown(
                f"<div style='background:#1e1e20;padding:1rem;border-radius:0.5rem;border:1px solid #333;'>{message}</div>",
                unsafe_allow_html=True
            )
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Bottom-docked input area (text + mic + send)
# =========================
# Safe clear prior to rendering the text input
if st.session_state.clear_text_next_run:
    st.session_state[TEXT_KEY] = ""
    st.session_state.clear_text_next_run = False

def _on_send_clicked():
    txt = st.session_state.get(TEXT_KEY, "")
    if txt and txt.strip():
        st.session_state.pending_user_input = txt.strip()
        st.session_state.just_sent = True
        st.session_state.clear_text_next_run = True
        st.session_state.scroll_to_bottom = True

st.markdown('<div class="input-container"><div class="input-inner">', unsafe_allow_html=True)
col_text, col_mic, col_send = st.columns([0.72, 0.12, 0.16])

with col_text:
    st.text_input(
        "Message",
        key=TEXT_KEY,
        placeholder="Message your assistant…",
        label_visibility="collapsed",
    )

with col_mic:
    # Real mic (high-contrast)
    audio_bytes = audio_recorder(
        text="",
        recording_color="#ff5a5f",
        neutral_color="#e6e6e6",
        icon_name="microphone",
        icon_size="2x",
        key=MIC_KEY,
    )

col_send.button("Send", use_container_width=True, key="send_button", on_click=_on_send_clicked)
st.markdown('</div></div>', unsafe_allow_html=True)

# ---- Only transcribe new audio (skip right after Send; skip dup clips)
def _digest(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

if audio_bytes and len(audio_bytes) > MIN_AUDIO_BYTES and not st.session_state.just_sent:
    dig = _digest(audio_bytes)
    if st.session_state.last_audio_digest != dig:
        try:
            transcript = transcribe_audio_bytes(
                audio_bytes=audio_bytes,
                api_key=os.getenv("OPENAI_API_KEY"),
                language=None
            ).strip()
            if transcript:
                st.session_state.pending_user_input = transcript
                st.session_state.last_audio_digest = dig
                st.session_state.scroll_to_bottom = True
        except Exception as e:
            st.error(f"Transcription error: {e}")

# Pull & reset pending input
user_input = st.session_state.pending_user_input
st.session_state.pending_user_input = None
st.session_state.just_sent = False

# =========================
# Handle user input (RAG + TTS, same behavior as before)
# =========================
if user_input:
    # Add user bubble
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.scroll_to_bottom = True

    # Generate assistant reply
    with st.chat_message("assistant", avatar="🤖"):
        msg_placeholder = st.empty()
        with st.spinner("Retrieving context and generating response..."):
            if is_ack(user_input):
                full_answer = fallback_general_response(user_input)
            else:
                query = user_input
                final_context = ""
                max_retries = 2
                context = ""
                for i in range(max_retries):
                    docs = retrieve_context(query)
                    if not docs:
                        full_answer = fallback_general_response(user_input)
                        break
                    else:
                        context = "\n".join(docs)

                    if verify_documents(user_input, context):
                        final_context = context
                        break
                    else:
                        query = get_missing_info_query(user_input, context)
                        time.sleep(1.0)

                if final_context:
                    full_answer = generate_response(user_input, final_context)
                else:
                    full_answer = full_answer if 'full_answer' in locals() else fallback_general_response(user_input)

        # Typewriter effect
        out_so_far = ""
        for ch in full_answer:
            out_so_far += ch
            msg_placeholder.markdown(
                f"<div style='background:#1e1e20;padding:1rem;border-radius:0.5rem;border:1px solid #333;'>{out_so_far}▌</div>",
                unsafe_allow_html=True
            )
            time.sleep(0.01)

        msg_placeholder.markdown(
            f"<div style='background:#1e1e20;padding:1rem;border-radius:0.5rem;border:1px solid #333;'>{full_answer}</div>",
            unsafe_allow_html=True
        )

        # 🔊 Speak reply if enabled
        if speak_replies and full_answer.strip():
            try:
                SimpleRateLimiter(0.75).wait()
                audio_out, mime = synthesize_speech_mp3(full_answer, lang="en")
                if audio_out:
                    st.audio(audio_out, format=mime, autoplay=True)
            except Exception as e:
                st.warning(f"Could not synthesize speech: {e}")

    # Persist assistant bubble
    st.session_state.chat_history.append(("assistant", full_answer))
    st.session_state.scroll_to_bottom = True
    st.rerun()

# =========================
# Auto-scroll to bottom after new content
# =========================
if st.session_state.scroll_to_bottom:
    st.session_state.scroll_to_bottom = False
    st.markdown("""
    <script>
        const el = document.getElementById('chat-container');
        if (el) { setTimeout(() => { el.scrollTop = el.scrollHeight; }, 100); }
    </script>
    """, unsafe_allow_html=True)
