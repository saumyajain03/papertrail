# ✦ papertrail
### *your document, your questions*

> A PDF-constrained conversational agent built with RAG — every answer grounded, every claim cited, every hallucination refused.

---

## ✦ what is this?

**papertrail** is a fully grounded conversational AI that lets you chat with any PDF document. It answers questions strictly from the content of your uploaded file — with page citations for every response — and explicitly refuses anything outside the document's scope.

---

## ✦ how it works

```
PDF upload → extract text (PyMuPDF)
          → chunk into pieces (LangChain RecursiveCharacterTextSplitter)
          → embed locally (Sentence Transformers / SBERT)
          → store in vector DB (ChromaDB)
          → query time: embed question → retrieve top 3 chunks
          → generate answer (Groq / Llama 3.1) with strict grounding prompt
          → return answer + page citations
```

No outside knowledge. No hallucinations. No guessing.

---

## ✦ tech stack

| Layer | Tool | Why |
|---|---|---|
| PDF Parsing | PyMuPDF | Fast, accurate, preserves page metadata |
| Chunking | LangChain `RecursiveCharacterTextSplitter` | Splits at paragraphs → sentences → words, preserving meaning |
| Embeddings | `sentence-transformers` (SBERT) | Runs fully local, no API key needed |
| Vector Store | ChromaDB | Persists to disk, production-grade, free |
| LLM | Groq (Llama 3.1) | Fast inference, free tier available |
| UI | Streamlit | Clean, Python-native, easy to deploy |

**Everything in this stack is free and runs locally.**

---

## ✦ getting started

### 1. clone the repo
```bash
git clone https://github.com/yourusername/papertrail.git
cd papertrail
```

### 2. install dependencies
```bash
pip install -r requirements.txt
```

### 3. set your Groq API key
Create a `.env` file in the root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at [console.groq.com](https://console.groq.com)

### 4. run the app
```bash
streamlit run app.py
```

---

## ✦ project structure

```
papertrail/
│
├── app.py                        # Main Streamlit UI
├── config.py                     # Chunk size, model names, thresholds
├── requirements.txt
├── .env                          # Your API keys (never commit this)
├── project_submission_eval.md    # test cases for testind the app                                 
│
├── ingestion/
│   ├── loader.py           # PDF text extraction (PyMuPDF)
│   ├── chunking.py         # RecursiveCharacterTextSplitter
│   └── embedding.py        # SBERT local embeddings
│
├── retrieval/
│   └── retriever.py        # ChromaDB build + cosine search
│
├── generation/
│   └── llm.py              # Groq LLM + grounding system prompt
│
└── tests/
    ├── sample.pdf           # Sample PDF for testing
```

---

## ✦ architecture decisions

### why recursive chunking over fixed-size?
Fixed-size chunking cuts sentences mid-meaning. `RecursiveCharacterTextSplitter` tries paragraph → sentence → word boundaries first, so chunks stay semantically coherent. Set at **1000 characters with 200 overlap** to avoid losing context at boundaries.

### why ChromaDB over manual cosine similarity?
Manual NumPy cosine similarity works for small PDFs but loads all embeddings into memory on every run. ChromaDB persists vectors to disk, meaning the PDF doesn't need to be re-embedded on restart. It scales to large documents and is the production-grade choice.

### why local SBERT embeddings?
Running embeddings locally with `sentence-transformers` means zero API cost, zero latency on the embedding step, and no data leaving your machine. The `all-MiniLM-L6-v2` model is small, fast, and accurate enough for document retrieval.

### why a refusal threshold?
If the top retrieved chunk has a similarity score below **0.25**, the system triggers an explicit refusal rather than attempting a low-confidence answer. This is the core anti-hallucination mechanism.

---

## ✦ the grounding system prompt

```
You are a document assistant. Answer questions only using the context provided.
For every statement you make, cite the page number in brackets like [Page X].
If the answer is not present in the context, respond with exactly:
"This information is not available in the provided document."
Never use outside knowledge. Never guess.
```

This prompt is non-negotiable — it is what makes the system grounded.

---

## ✦ test cases

### valid queries (should answer with citations)
1. **"What is the primary architecture proposed in this paper?"** 
   * **Expected:** The bot will explain the Transformer model, relying on attention mechanisms instead of RNNs/CNNs, and cite [Page 1] or [Page 2].
2. **"How does Multi-Head Attention work?"** 
   * **Expected:** It will explain that it projects queries, keys, and values $h$ times in parallel, then concatenates them, citing [Page 4] or [Page 5].
3. **"What optimizer and learning rate schedule was used for training?"** 
   * **Expected:** It will explicitly mention the Adam optimizer and the specific warm-up steps formula used, citing [Page 5.3] or section specific pages.
4. **"Summarize the BLEU scores achieved on the WMT 2014 English-to-French Translation task."** 
   * **Expected:** It will extract the 41.0 BLEU score and cite [Page 8] or [Page 9].
5. **"¿Cuál es la conclusión de este documento?"** *(Bonus Multilingual Test)* 
   * **Expected:** It will fetch the English conclusion chunks and translate its accurate response into perfect Spanish, citing [Page 10].
6. **"Is paper ka primary architecture kya describe karta hai?"** *(Bonus Hinglish Test)*
   * **Expected:** It seamlessly natively understands the blended Hindi-English syntax, cross-references it against the English embeddings, and thoughtfully replies back naturally in Hinglish citing [Page 1].

### invalid / out-of-scope queries (should refuse)
*Please test these queries to evaluate our strict anti-hallucination barriers.*
1. **"What is the current stock price of NVIDIA?"**
   * **Expected:** Even though NVIDIA GPUs are mentioned in the text for training, the bot will recognize stock prices aren't present and present the visual refusal box.
2. **"Give me a recipe for chocolate chip cookies."**
   * **Expected:** Immediate refusal. The prompt completely lacks semantic relevance to the embedded paper.
3. **"Who won the World Cup in 2022?"**
   * **Expected:** Immediate refusal. No external knowledge allowed. 

---

## ✦ bonus: multilingual support

papertrail handles non-English queries out of the box via Llama 3.1's multilingual capability. Ask a question in Hindi, French, or Spanish — the system retrieves the same grounded chunks and responds in your language.

*Tested with Hindi queries on English PDFs — grounding and citations remain intact.*

---

## ✦ known trade-offs

- **Scanned PDFs** — PyMuPDF cannot extract text from image-only scans. OCR support (e.g. pytesseract) would be needed for those.
- **Very large PDFs (500+ pages)** — ChromaDB handles this fine but SBERT embedding time increases linearly. A progress bar is shown during ingestion.
- **Table data** — tabular content in PDFs often loses structure during text extraction. Complex tables may not retrieve accurately.

---

## ✦ evaluation criteria mapping

| Criterion | How papertrail addresses it |
|---|---|
| Accuracy relative to source | Strict system prompt + source-only context injection |
| Robustness against hallucination | Refusal threshold at similarity score < 0.25 |
| Quality of refusal | Distinct UI treatment + explicit refusal message |
| Retrieval and grounding quality | RecursiveChunking + ChromaDB + SBERT |
| Observability | Chunk count, page count, score shown in sidebar |

---

## ✦ contact

Built by **Saumya Jain**

---

*papertrail — because every answer should leave a trail.*

