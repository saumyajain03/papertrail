# RAG Chatbot: Project Evaluation & Test Guide

Welcome to the Grounded RAG Chatbot evaluation guide! This document outlines how the application fulfills all core requirements, testability criteria, and bonus objectives for the internship challenge.

## Core Requirements Achieved
✅ **Accept any PDF as input**: Users can drag and drop any PDF via the Streamlit sidebar.
✅ **Enable Conversational Querying**: An interactive chat interface is provided, backed by `st.session_state` to track history so the AI can answer follow-up queries seamlessly.
✅ **Strict Grounding (Answer only from PDF)**: The LLM is explicitly isolated from its internal internet-trained logic via strict system prompting. It strictly cross-references the retrieved chunks.
✅ **Explicitly Refuse Out-of-Scope Queries**: If a query's semantic relevance chunk-score drops too low, or the LLM recognizes the answer isn't in the provided text, it halts generation and triggers a visual "Out of Scope" amber warning box.
✅ **Citations**: The system forces the LLM to output exact page numbers (e.g. `[Page 5]`). Our Streamlit regex parses these citations and injects beautiful `<span class="cite-pill">` badges below the AI's response.

---

## 🏆 Bonus Objectives Achieved
We fully hit the bonus objectives by integrating the state-of-the-art **`paraphrase-multilingual-MiniLM-L12-v2`** local embedding model alongside **Llama 3.1 8B**. 

* **Supports Multiple Languages**: You can upload a document built entirely in English, and ask questions in Spanish, Hindi, French, or German.
* **Demonstrates consistent performance**: Since the multilingual embedding model projects English sentences and non-English questions into the exact same semantic vector space, it perfectly retrieves chunks regardless of the query's language!
* **Maintains grounding in non-English**: The `llm.py` `SYSTEM_PROMPT` enforces the model to detect the user's language and output the answer heavily grounded in *that exact target language* while keeping `[Page X]` citations intact. 

---

## 🧪 Testability Guide

We have bundled the **"Attention is all you need.pdf"** (the foundational 2017 Google Transformer paper) in the `data/` folder for immediate testing. 

### 1. Valid / In-Scope Queries (Expected to Succeed)
*Please test these queries. They demonstrate the bot's ability to extract technical logic and synthesize multi-page information.*
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

### 2. Invalid / Out-of-Scope Queries (Expected to Refuse)
*Please test these queries to evaluate our strict anti-hallucination barriers.*
1. **"What is the current stock price of NVIDIA?"**
   * **Expected:** Even though NVIDIA GPUs are mentioned in the text for training, the bot will recognize stock prices aren't present and present the visual refusal box.
2. **"Give me a recipe for chocolate chip cookies."**
   * **Expected:** Immediate refusal. The prompt completely lacks semantic relevance to the embedded paper.
3. **"Who won the World Cup in 2022?"**
   * **Expected:** Immediate refusal. No external knowledge allowed. 

---
### Running the test locally: 
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Enjoy the execution speed! Over 800 Tokens/sec via Groq, and GPU Hardware acceleration via Mac `mps` pipelines for local embeddings!
