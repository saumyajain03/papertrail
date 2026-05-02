import streamlit as st
import os
import tempfile
import re
import config
from ingestion import loader, chunking, embedding
from retrieval import retriever
from generation import llm

st.set_page_config(page_title="papertrail ✦", layout="wide", page_icon="✦")

custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&family=Playfair+Display:ital,wght@1,400&display=swap');

/* ── Reset & Root ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --rose:      #f2c4ce;
    --rose-deep: #d4899a;
    --blush:     #fdf0f3;
    --cream:     #fdf8f5;
    --linen:     #f7efe8;
    --warm-grey: #b8a9a3;
    --ink:       #3b2f2f;
    --ink-light: #7a6560;
    --sage:      #b5c4b1;
    --sage-deep: #7a9474;
    --lavender:  #c9c0d3;
    --gold:      #d4a96a;
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--cream) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--ink);
}

/* ── Hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    min-width: 280px !important;
    max-width: 280px !important;
    background: linear-gradient(180deg, #fff5f7 0%, var(--linen) 100%) !important;
    border-right: 1px solid #f0dde3 !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

/* ── Sidebar branding ── */
.brand-header {
    background: linear-gradient(135deg, #f7dce3 0%, #ede0f5 100%);
    padding: 28px 24px 22px;
    border-bottom: 1px solid #f0dde3;
    margin-bottom: 0;
}
.brand-wordmark {
    font-family: 'DM Serif Display', serif;
    font-size: 26px;
    color: var(--ink);
    letter-spacing: -0.5px;
    line-height: 1;
}
.brand-wordmark span { color: var(--rose-deep); }
.brand-tagline {
    font-size: 11px;
    color: var(--ink-light);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 5px;
    font-weight: 300;
}

/* ── Upload zone ── */
.upload-label {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--ink-light);
    margin: 20px 0 8px 0;
    padding: 0 24px;
    display: block;
}
[data-testid="stFileUploader"] {
    padding: 0 16px;
}
[data-testid="stFileUploader"] > div {
margin: 32px 16px 16px;
    border: none !important;
    border-radius: 20px !important;
    background: var(--rose) !important;
    padding: 16px !important;
    transition: all 0.2s;
    box-shadow: 0 4px 12px rgba(242, 196, 206, 0.4);
}
[data-testid="stFileUploader"] > div:hover {
    background: var(--rose-deep) !important;
}
[data-testid="stFileUploader"] label p { font-size: 12px !important; color: var(--ink) !important; font-weight: 500; }

/* ── PDF info card ── */
.pdf-card {
    background: var(--rose);
    border: none;
    border-radius: 14px;
    padding: 14px 16px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.pdf-thumb {
    width: 36px; height: 42px;
    background: rgba(255, 255, 255, 0.4);
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 10px; font-weight: 600;
    color: var(--ink); flex-shrink: 0;
    letter-spacing: 0.05em;
}
.pdf-meta-name {
    font-size: 12px; font-weight: 500; color: var(--ink);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 160px;
}
.pdf-meta-sub { font-size: 10px; color: var(--ink-light); margin-top: 2px; }
.pdf-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--sage-deep); margin-left: auto; flex-shrink: 0; }

/* ── Stat grid ── */
.stats-wrap { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; padding: 14px 16px 0; }
.stat-tile {
    background: white;
    border: 1px solid #f0dde3;
    border-radius: 12px;
    padding: 12px 10px;
    text-align: center;
}
.stat-tile .val {
    font-family: 'DM Serif Display', serif;
    font-size: 22px; color: var(--ink);
    line-height: 1;
}
.stat-tile .lbl {
    font-size: 9px; font-weight: 500; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--warm-grey); margin-top: 3px;
}

/* ── Divider ── */
.soft-divider {
    height: 1px; background: linear-gradient(90deg, transparent, #f0dde3, transparent);
    margin: 16px 0;
}

/* ── Recent questions ── */
.recent-label {
    font-size: 9px; font-weight: 500; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--warm-grey);
    padding: 0 24px; display: block; margin-bottom: 6px;
}
.recent-item {
    padding: 7px 24px;
    font-size: 12px; color: var(--ink-light);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    cursor: pointer;
    border-left: 2px solid transparent;
    transition: all 0.15s;
}
.recent-item:hover { color: var(--rose-deep); border-left-color: var(--rose); }

/* ── Main area ── */
.main-header {
    padding: 32px 40px 20px;
    border-bottom: 1px solid #f0dde3;
    display: flex; align-items: center; gap: 12px;
}
.main-title {
    font-family: 'DM Serif Display', serif;
    font-size: 28px; color: var(--ink); letter-spacing: -0.5px;
}
.main-title em { font-style: italic; color: var(--rose-deep); }

/* ── Badges ── */
.badge {
    display: inline-block;
    font-size: 10px; font-weight: 500; letter-spacing: 0.08em;
    text-transform: uppercase; padding: 4px 10px;
    border-radius: 20px; border: 1px solid;
}
.badge-grounded { background: #eef5ec; color: var(--sage-deep); border-color: #cddeca; }
.badge-rag { background: #f5f0fa; color: #8b6fb5; border-color: #ddd0ee; }
.badge-live { background: #fff5f7; color: var(--rose-deep); border-color: #f0c5d0; }

/* ── Empty state ── */
.empty-state {
    text-align: center; padding: 60px 40px;
}
.empty-icon {
    font-size: 48px; margin-bottom: 16px; opacity: 0.4;
}
.empty-title {
    font-family: 'DM Serif Display', serif;
    font-size: 22px; color: var(--ink); margin-bottom: 8px;
}
.empty-sub { font-size: 13px; color: var(--warm-grey); line-height: 1.7; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 4px 0 !important;
}

/* User message bubble */
[data-testid="stChatMessage"][data-testid*="user"] .stMarkdown p,
.user-bubble {
    background: linear-gradient(135deg, #f7dce3 0%, #ede0f5 100%);
    border-radius: 18px 18px 4px 18px;
    padding: 11px 16px;
    font-size: 14px;
    color: var(--ink);
    display: inline-block;
    max-width: 80%;
    line-height: 1.6;
}

/* AI avatar area */
[data-testid="stChatMessage"] .stAvatar {
    border-radius: 50% !important;
}

/* ── Citation pills ── */
.citations-wrap { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 10px; }
.cite-pill {
    display: inline-flex; align-items: center; gap: 4px;
    background: #fff5f7;
    border: 1px solid #f0c5d0;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 11px; font-weight: 500;
    color: var(--rose-deep);
    cursor: default;
}
.cite-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--rose); }

/* ── Refusal box ── */
.refusal-box {
    background: linear-gradient(135deg, #fffbf0, #fff8ec);
    border: 1px solid #f5d9a0;
    border-left: 3px solid var(--gold);
    border-radius: 12px;
    padding: 14px 18px;
    font-size: 13px; color: #8a6530;
    line-height: 1.6;
}
.refusal-box .refusal-icon { font-size: 16px; margin-right: 6px; }

/* ── AI answer text ── */
.answer-text {
    font-size: 14px; line-height: 1.75;
    color: var(--ink);
}

/* ── Thinking indicator ── */
.thinking-wrap {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 0;
}
.thinking-text { font-size: 12px; color: var(--warm-grey); font-style: italic; }
.dots { display: flex; gap: 4px; }
.dot {
    width: 5px; height: 5px; border-radius: 50%;
    background: var(--rose);
    animation: blink 1.3s infinite;
}
.dot:nth-child(2) { animation-delay: 0.2s; }
.dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink { 0%,80%,100% { opacity: 0.2; } 40% { opacity: 1; } }

/* ── Suggestion chips ── */
.chip-row { display: flex; gap: 8px; flex-wrap: wrap; padding: 10px 0 4px; }

div.stButton > button {
    background: white !important;
    border: 1px solid #f0dde3 !important;
    border-radius: 20px !important;
    padding: 6px 16px !important;
    font-size: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--ink-light) !important;
    font-weight: 400 !important;
    transition: all 0.2s !important;
    box-shadow: none !important;
}
div.stButton > button:hover {
    background: #fff5f7 !important;
    border-color: var(--rose) !important;
    color: var(--rose-deep) !important;
}

/* ── Chat input ── */
[data-testid="stBottom"] {
    background: transparent !important;
}
[data-testid="stBottom"] > div {
    background: transparent !important;
}
[data-testid="stChatInput"] {
    background: transparent !important;
}
[data-testid="stBottomBlockContainer"] {
    background: transparent !important;
}
[data-testid="stChatInputContainer"] {
    background: var(--rose) !important;
    border: none !important;
    border-radius: 50px !important; /* Creates the curved pill bar look */
    padding: 6px 16px !important;
    box-shadow: 0 4px 16px rgba(212, 137, 154, 0.2) !important;
}
[data-testid="stChatInputContainer"] textarea {
    background: transparent !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    color: var(--ink) !important;
}
[data-testid="stChatInputContainer"] button {
    background: linear-gradient(135deg, var(--rose-deep), #c46fa0) !important;
    border-radius: 50% !important;
    border: none !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #f0c5d0; border-radius: 4px; }

/* ── Spinner ── */
[data-testid="stSpinner"] p { font-size: 12px !important; color: var(--warm-grey) !important; }

/* ── Warning / info streamlit ── */
[data-testid="stAlert"] { border-radius: 12px !important; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ─── State ───────────────────────────────────────────────────────────────────
for k, v in {
    "collection": None,
    "chat_history": [],
    "internal_history": [],
    "stats": {"filename": None, "pages": 0, "chunks": 0},
    "suggestion_clicked": None,
    "last_retrieved_chunks": [],
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Helpers ─────────────────────────────────────────────────────────────────
def render_citations(text: str):
    matches = re.findall(r'\[Pages?\s*([0-9\.\-\,]+)\]', text, re.IGNORECASE)
    clean   = re.sub(r'\[Pages?\s*[0-9\.\-\,]+\]', '', text, flags=re.IGNORECASE).strip()
    pills   = ""
    if matches:
        for p in sorted(set(matches)):
            pills += f'<span class="cite-pill"><span class="cite-dot"></span>pg {p}</span>'
    return clean, pills

def is_refusal(text: str) -> bool:
    return "This information is not available in the provided document" in text

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div class="brand-header">
            <div class="brand-wordmark">paper<span>trail</span> ✦</div>
            <div class="brand-tagline">your document, your questions</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="upload-label">drop a pdf</span>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")

    if uploaded_file and st.session_state.stats["filename"] != uploaded_file.name:
        with st.spinner("reading your document…"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            try:
                pages          = loader.load_pdf(tmp_path)
                chunks         = chunking.chunk_text(pages, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
                embedded_chunks= embedding.embed_chunks(chunks, config.EMBEDDING_MODEL)
                collection     = retriever.build_vectorstore(embedded_chunks)
                st.session_state.collection     = collection
                st.session_state.stats          = {"filename": uploaded_file.name, "pages": len(pages), "chunks": len(chunks)}
                st.session_state.chat_history   = []
                st.session_state.internal_history = []
                st.session_state.last_retrieved_chunks = []
                os.remove(tmp_path)
                st.rerun()
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
                os.remove(tmp_path)

    if st.session_state.stats["filename"]:
        name = st.session_state.stats["filename"]
        st.markdown(f"""
            <div class="pdf-card">
                <div class="pdf-thumb">PDF</div>
                <div>
                    <div class="pdf-meta-name" title="{name}">{name}</div>
                    <div class="pdf-meta-sub">ready</div>
                </div>
                <div class="pdf-dot"></div>
            </div>
        """, unsafe_allow_html=True)

        user_qs = [m["content"] for m in st.session_state.chat_history if m["role"] == "user"]
        if user_qs:
            st.markdown('<span class="recent-label" style="margin-top: 10px;">recent questions</span>', unsafe_allow_html=True)
            for q in reversed(user_qs[-4:]):
                st.markdown(f'<div class="recent-item">✦ {q}</div>', unsafe_allow_html=True)
            st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

        st.markdown(f"""
            <div style="padding: 0 24px; font-size: 11px; color: var(--warm-grey); text-transform: uppercase; letter-spacing: 0.1em; display: flex; align-items: center; justify-content: space-between;">
                <span>indexing status</span> 
                <span style="color: var(--rose-deep); font-weight: 600; background: #fff5f7; padding: 2px 8px; border-radius: 12px;">{st.session_state.stats['pages']} Pages Ready</span>
            </div>
        """, unsafe_allow_html=True)

# ─── Main ─────────────────────────────────────────────────────────────────────
st.markdown("""
    <div class="main-header">
        <div class="main-title"><em>ask</em> your document</div>
        <span class="badge badge-grounded">grounded</span>
        <span class="badge badge-rag">rag</span>
        <span class="badge badge-live">live</span>
    </div>
""", unsafe_allow_html=True)

# Empty state
if not st.session_state.collection:
    st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">✦</div>
            <div class="empty-title">upload a pdf to begin</div>
            <div class="empty-sub">
                drop any document in the sidebar and start asking questions.<br>
                every answer is grounded — with page citations, always.
            </div>
        </div>
    """, unsafe_allow_html=True)
else:
    # Chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            with st.chat_message("assistant"):
                if is_refusal(msg["content"]):
                    st.markdown(f"""
                        <div class="refusal-box">
                            <span class="refusal-icon">✧</span>
                            this one's outside the document — i can only speak to what's written here.
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    clean, pills = render_citations(msg["content"])
                    st.markdown(f'<div class="answer-text">{clean}</div>', unsafe_allow_html=True)
                    if pills:
                        st.markdown(f'<div class="citations-wrap">{pills}</div>', unsafe_allow_html=True)

    # Suggestion chips
    st.markdown('<div class="chip-row">', unsafe_allow_html=True)
    c1, c2, c3, _ = st.columns([2, 2, 2.5, 3])
    chips = ["✦ summarise", "✦ key findings", "✦ conclusions"]
    vals  = ["Summarise the document", "What are the key findings?", "What is the conclusion?"]
    for col, chip, val in zip([c1, c2, c3], chips, vals):
        if col.button(chip):
            st.session_state.suggestion_clicked = val
    st.markdown('</div>', unsafe_allow_html=True)

# ─── Input & response ─────────────────────────────────────────────────────────
prompt = st.chat_input("ask anything about your document…")
if st.session_state.suggestion_clicked:
    prompt = st.session_state.suggestion_clicked
    st.session_state.suggestion_clicked = None

if prompt:
    if not st.session_state.collection:
        st.warning("upload a pdf first, then we can chat ✦")
    else:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(f'<div class="user-bubble">{prompt}</div>', unsafe_allow_html=True)

        with st.chat_message("assistant"):
            try:
                # Agentic Query Expansion
                with st.spinner("agentic reasoning: optimizing query…"):
                    optimized_query = llm.rewrite_query(prompt, st.session_state.internal_history)
                    # Show the user the agent at work!
                    if optimized_query.lower() != prompt.lower():
                        st.markdown(f'<div style="font-size: 11px; color: #b8a9a3; margin-bottom: 8px;">✦ optimized search: {optimized_query}</div>', unsafe_allow_html=True)
                
                with st.spinner("finding the right pages…"):
                    top_chunks  = retriever.search_query(optimized_query, st.session_state.collection, top_k=7)
                    best_score  = top_chunks[0]["score"] if top_chunks else 0
                    THRESHOLD   = 0.25
                    if best_score < THRESHOLD and st.session_state.last_retrieved_chunks:
                        top_chunks = st.session_state.last_retrieved_chunks
                    elif best_score >= THRESHOLD:
                        st.session_state.last_retrieved_chunks = top_chunks
                    answer = llm.generate_answer(prompt, top_chunks, st.session_state.internal_history)

                if is_refusal(answer):
                    st.markdown("""
                        <div class="refusal-box">
                            <span class="refusal-icon">✧</span>
                            this one's outside the document — i can only speak to what's written here.
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    clean, pills = render_citations(answer)
                    st.markdown(f'<div class="answer-text">{clean}</div>', unsafe_allow_html=True)
                    if pills:
                        st.markdown(f'<div class="citations-wrap">{pills}</div>', unsafe_allow_html=True)

                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.session_state.internal_history.extend([
                    {"role": "user",      "content": prompt},
                    {"role": "assistant", "content": answer},
                ])
            except Exception as e:
                st.error(f"Error answering question: {str(e)}")