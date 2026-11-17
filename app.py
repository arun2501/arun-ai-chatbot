"""
app.py
Flask Web Application for Arunkumar K R's Personal AI Chatbot.
Provides REST endpoints for chat prediction, RAG synchronization, and portfolio embedding.
"""

import os
import sys
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load local .env if present
load_dotenv()

# Force UTF-8 on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from chat import get_detailed_response, sync_knowledge_base, rag_engine

app = Flask(__name__)
# Enable CORS for all routes so Netlify portfolio can call API
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/")
def index_get():
    """Render the primary chatbot interface."""
    return render_template("base.html")

@app.route("/embed")
def embed_view():
    """Render embeddable widget view for iframe injection."""
    return render_template("base.html", embedded=True)

@app.route("/predict", methods=["POST"])
def predict():
    """Process user message through RAG retrieval and dual-engine synthesizer."""
    try:
        data = request.get_json(silent=True) or {}
        text = data.get("message", "").strip()

        if not text:
            return jsonify({
                "answer": "Please ask a question about Arunkumar's background, education, projects, or experience!",
                "engine": "System",
                "sources": [],
                "suggested_chips": ["🚀 Top Projects", "💼 Work Experience", "🎓 Education & CGPA", "📞 Contact & Hire"]
            })

        result = get_detailed_response(text)
        return jsonify(result)
    except Exception as e:
        print(f"[app.py] Error in /predict: {e}")
        return jsonify({
            "answer": "I apologize, but I encountered an unexpected error. Please try asking again or feel free to contact Arun directly at krarun2504@gmail.com.",
            "engine": "Fallback",
            "sources": [],
            "suggested_chips": ["📞 How to Contact Arunkumar", "📄 Download CV"]
        }), 500

@app.route("/api/sync", methods=["GET", "POST"])
def sync_portfolio_api():
    """Trigger on-demand live synchronization from Arunkumar's Netlify portfolio."""
    try:
        reindex_info = sync_knowledge_base()
        return jsonify({
            "status": "success",
            "message": "Portfolio data fetched and RAG knowledge base updated successfully!",
            "details": reindex_info
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to sync portfolio: {str(e)}"
        }), 500

@app.route("/api/status", methods=["GET"])
def status_api():
    """Return health check, active RAG stats, and configured engines."""
    groq_configured = bool(os.environ.get("GROQ_API_KEY", "").strip())
    gemini_configured = bool(os.environ.get("GEMINI_API_KEY", "").strip())
    return jsonify({
        "status": "online",
        "bot_name": "Arun AI",
        "total_knowledge_chunks": len(rag_engine.chunks),
        "groq_configured": groq_configured,
        "gemini_configured": gemini_configured,
        "default_engine": "Local Neural RAG Synthesizer (Zero Paid API Keys Required)" if not (groq_configured or gemini_configured) else "Free Cloud LLM"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("DEBUG", "False").lower() in ["true", "1"]
    print(f"\n=======================================================")
    print(f"  Arunkumar K R Personal AI Chatbot Server Starting")
    print(f"  Local URL: http://localhost:{port}")
    print(f"  RAG Chunks Loaded: {len(rag_engine.chunks)}")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
