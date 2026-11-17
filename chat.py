"""
chat.py
Core Chat Controller for Arunkumar K R's Personal Chatbot.
Integrates RAGEngine with ResponseSynthesizer for full-context, multi-angle responses.
Zero paid API keys needed, with optional Free Cloud LLM support.
"""

import os
import sys
from rag_engine import RAGEngine
from synthesizer import ResponseSynthesizer

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

bot_name = "Arun AI"
rag_engine = RAGEngine()
synthesizer = ResponseSynthesizer()

def get_detailed_response(msg: str, groq_key: str = None, gemini_key: str = None) -> dict:
    """
    Primary interface for RAG chat.
    Returns:
        dict: {
            'answer': str,
            'engine': str,
            'sources': list[str],
            'suggested_chips': list[str]
        }
    """
    if not msg or not msg.strip():
        return {
            'answer': "Please ask a question about Arunkumar's background, education, projects, or experience!",
            'engine': 'System',
            'sources': [],
            'suggested_chips': ["🚀 Top Projects", "💼 Work Experience", "🎓 Education & CGPA", "📞 Contact & Hire"]
        }

    # Update keys if supplied dynamically
    if groq_key or gemini_key:
        synthesizer.update_keys(groq_key=groq_key, gemini_key=gemini_key)

    # 1. Retrieve top-scoring semantic chunks
    retrieved_chunks = rag_engine.retrieve(msg, top_k=4)

    # 2. Synthesize multi-angle response
    result = synthesizer.synthesize(msg, retrieved_chunks)
    return result

def get_response(msg: str) -> str:
    """Legacy helper returning plain string response."""
    res = get_detailed_response(msg)
    return res['answer']

def sync_knowledge_base():
    """Trigger live sync from portfolio and rebuild RAG index."""
    try:
        from sync_portfolio import fetch_portfolio_data
        fetch_portfolio_data()
    except Exception as e:
        print(f"[chat.py] Portfolio sync notice: {e}")
    return rag_engine.reindex()

if __name__ == "__main__":
    print(f"==================================================")
    print(f"  Let's chat with {bot_name}! (Type 'quit' or 'exit' to end)")
    print(f"  Ask anything about Arunkumar's Projects, Skills,")
    print(f"  Internships, IEEE Papers, CGPA, or Contact Info.")
    print(f"==================================================\n")

    while True:
        try:
            sentence = input("You: ").strip()
            if sentence.lower() in ["quit", "exit", "q"]:
                print(f"{bot_name}: Thank you for chatting! Best wishes!")
                break
            if not sentence:
                continue

            resp = get_detailed_response(sentence)
            print(f"\n{bot_name} [{resp['engine']}]:\n{resp['answer']}\n")
            if resp.get('suggested_chips'):
                print(f"Suggestions: {' | '.join(resp['suggested_chips'])}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat. Have a great day!")
            break
