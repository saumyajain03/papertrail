"""
Answer generation using Groq API (free tier).
Supports conversation memory — passes the last N turns of chat history.
"""

from openai import OpenAI
import config

SYSTEM_PROMPT = (
    "You are a document assistant. Answer questions only using the context provided. "
    "For every statement you make, cite the page number in brackets like [Page X]. "
    "If the answer is not present in the context, respond with exactly: "
    "This information is not available in the provided document. "
    "Never use outside knowledge. "
    "IMPORTANT: You must detect the language of the user's question and respond in the exact same language. "
    "If the question is a mix of languages (e.g., English and Hindi/Hinglish), process it seamlessly and reply in a natural, similarly mixed conversational style or the dominant language."
)

MAX_HISTORY_TURNS = 6  # Keep last 6 messages (3 user + 3 assistant)


def generate_answer(query: str, retrieved_chunks: list, conversation_history: list) -> str:
    """Send retrieved chunks + conversation history + query to Groq."""

    if not retrieved_chunks:
        return "No relevant context was found to answer this question."

    # Format retrieved chunks as context block
    context_parts = []
    for chunk in retrieved_chunks:
        context_parts.append(f"[Page {chunk['page_number']}]:\n{chunk['text']}")
    context_block = "\n\n---\n\n".join(context_parts)

    # The current user message includes fresh context + current question
    user_message = (
        f"Context from the document:\n\n{context_block}\n\n"
        f"---\n\nQuestion: {query}"
    )

    # Cap history to the last MAX_HISTORY_TURNS messages
    trimmed_history = conversation_history[-MAX_HISTORY_TURNS:]

    # Build message list: system + past history + current question
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(trimmed_history)
    messages.append({"role": "user", "content": user_message})

    client = OpenAI(
        api_key=config.GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )

    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            max_tokens=1024,
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {e}"

def rewrite_query(query: str, chat_history: list) -> str:
    """Agentic RAG: Expand user query for dense vector retrieval."""
    system_prompt = (
        "You are an expert search query optimizer for a document retrieval system. "
        "Take the user's question and rewrite it into a dense, keyword-rich search query "
        "that will perfectly match the terminology found inside a formal document.\n\n"
        "EXAMPLES:\n"
        "- User: 'What is the conclusion?' -> Output: 'Conclusion, final results, concluding remarks, future work, findings summary'\n"
        "- User: 'Summarize the paper' -> Output: 'Abstract, introduction, overview, main contributions, summary framework'\n"
        "- User: 'How does it work?' -> Output: 'Mechanism, architecture, methodology, implementation details, process'\n\n"
        "RULES:\n"
        "1. Output ONLY the optimized search string. No explanations, no quotes.\n"
        "2. Factor in the conversation history if the query says 'it' or 'they'.\n"
        "3. Limit to 15 words maximum."
    )
    
    trimmed_history = chat_history[-MAX_HISTORY_TURNS:]
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(trimmed_history)
    messages.append({"role": "user", "content": query})

    client = OpenAI(api_key=config.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    
    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            max_tokens=50,
            messages=messages
        )
        # Strip potential quotes safely
        return response.choices[0].message.content.strip('"\'')
    except Exception:
        return query  # Fallback to the original query if anything fails
