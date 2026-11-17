# 🤖 Arunkumar K R - Personal AI Chatbot (GPT-Style with RAG)

A personal AI chatbot representing **Arunkumar K R** (Full Stack Developer, Data Analyst, and AI/ML Engineer). Built with a **Retrieval-Augmented Generation (RAG)** architecture that ingests Arunkumar's official resume, academic background, internships, projects, IEEE publications, and live portfolio data.

> **💡 Zero Paid API Keys Required!**  
> Works 100% out of the box with the built-in **Local Neural RAG Synthesizer**.  
> Optionally connects to **100% Free Cloud LLMs** (Groq LLaMA 3.3 70B or Google Gemini Flash) without needing a credit card.

---

## 🚀 Key Features

- **Multi-Angle Intelligence**:
  - Never skips, fails, or silently rejects queries.
  - Understands questions across all angles: education (8.84 CGPA at Velammal College), work experience at Elysium Technologies & Reccsar, the national award-winning IoT Pulse Nebulizer (APN), 2 IEEE research publications, tech stack (Java, Spring Boot, React.js, ML), and direct contact channels.
- **RAG Architecture with Live Synchronization**:
  - Semantic knowledge chunking with BM25 / sublinear TF-IDF retrieval + exact entity boosting.
  - **Live Portfolio Sync**: Includes `sync_portfolio.py` and a one-click `/api/sync` endpoint that live-scrapes [https://krarun-portfolio.netlify.app/](https://krarun-portfolio.netlify.app/) to refresh the knowledge base dynamically without server restarts.
- **Dual-Engine Generation**:
  1. **Built-in Local Synthesizer** (Default): Fast, offline, zero API keys needed, generates natural first-person conversational responses with markdown formatting, links, and interactive follow-up chips.
  2. **Free Cloud LLM (Optional)**: Supports **Groq Cloud** (LLaMA 3.3 70B) and **Google Gemini 2.0 / 1.5 Flash** (1,500 free queries/day, no credit card required).
- **Industry-Grade Modern UI**:
  - Clean ChatGPT / Intercom style aesthetic with verified badge and online indicator.
  - Interactive quick-start suggestion chips.
  - Markdown formatting with clickable links to GitHub repos, IEEE papers, and live demos.
  - 🎤 **Voice Input (Speech-to-Text)** via native browser Web Speech API.
  - 🔊 **Read Aloud (Text-to-Speech)** voice synthesis.
  - 📋 One-click copy answer to clipboard.
  - 📄 Direct CV / Resume download button.
  - Mobile & desktop responsive.

---

## 📂 Project Structure

```text
smart-chatbot-main/
│
├── knowledge/                     # Multi-domain RAG Knowledge Base
│   ├── profile.json               # Bio, contact, social links, resume URL
│   ├── education.json             # Velammal College (8.84 CGPA), Schooling
│   ├── skills.json                # Java, Spring Boot, React, Python, ML, Figma
│   ├── experience.json            # Elysium Technologies, Reccsar
│   ├── projects.json              # APN Nebulizer, ML Smart e-PDS, Fruit MMCNN, etc.
│   ├── publications.json          # IEEE Xplore published research papers
│   ├── certifications.json        # Wipro, IBM, Microsoft, TCS iON, PMI
│   ├── faq.json                   # Curated recruiter & visitor FAQs
│   └── live_portfolio.json        # Auto-synced data from Netlify portfolio
│
├── rag_engine.py                  # Semantic Chunking & BM25/TF-IDF Retrieval Engine
├── synthesizer.py                 # Dual-Engine Synthesizer (Local + Free Groq/Gemini LLM)
├── sync_portfolio.py              # Live Netlify Portfolio Web Scraper & Sync
├── chat.py                        # Core Chatbot Controller & Terminal CLI
├── app.py                         # Flask Web Server with REST API & CORS
├── requirements.txt               # Lightweight production dependencies
├── .env.example                   # Free API key configuration template
├── static/                        # Frontend Assets (CSS, JS, Images)
│   ├── app.js                     # Interactive client logic, STT, TTS, Toast
│   ├── style.css                  # Modern responsive GPT styling
│   └── images/                    # Arun's profile avatar
├── templates/
│   └── base.html                  # Full-featured web interface
└── standalone-frontend/           # Standalone static version for hosting on Netlify
```

---

## ⚡ Quick Start (Local Setup)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Sync with Live Portfolio
To scrape and index the latest changes from your live portfolio:
```bash
python sync_portfolio.py
```

### 3. Run the Chatbot Server
```bash
python app.py
```

Open your browser and navigate to:
```
http://localhost:5000
```

---

## 🔑 How to Use 100% Free Cloud LLM (Optional)

You **do not** need any paid API keys. If you want the bot to use an ultra-smart Cloud LLM, you can obtain a 100% free key from either provider:

### Option A: Groq Cloud (Recommended - Ultra Fast LLaMA 3.3 70B)
1. Go to [https://console.groq.com/keys](https://console.groq.com/keys)
2. Sign in with GitHub/Google and create a free API key.
3. Add it to your `.env` file:
   ```env
   GROQ_API_KEY=gsk_your_free_key_here
   ```
   *(Or click the ⚙️ Settings icon directly in the chatbot UI and paste it!)*

### Option B: Google Gemini (1,500 Free Requests/Day)
1. Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Generate your free Gemini API key (no credit card required).
3. Add it to `.env`:
   ```env
   GEMINI_API_KEY=AIzaSy_your_free_key_here
   ```

---

## 🌐 How to Embed into Your Netlify Portfolio

To add this chatbot directly into your live website (`https://krarun-portfolio.netlify.app/`), choose either method:

### Method 1: Floating Iframe Widget
Add this snippet just before the closing `</body>` tag of your portfolio `index.html`:

```html
<!-- Arunkumar AI Chatbot Widget -->
<div id="arun-ai-widget" style="position: fixed; bottom: 20px; right: 20px; z-index: 99999;">
  <button id="toggle-chat-btn" style="width: 60px; height: 60px; border-radius: 50%; background: #2563eb; color: #fff; border: none; box-shadow: 0 4px 16px rgba(37,99,235,0.4); cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 26px;">
    💬
  </button>
  <div id="chat-frame-box" style="display: none; position: absolute; bottom: 75px; right: 0; width: 380px; height: 560px; max-width: 90vw; max-height: 80vh; border-radius: 16px; overflow: hidden; box-shadow: 0 12px 36px rgba(0,0,0,0.25);">
    <iframe src="http://localhost:5000" style="width: 100%; height: 100%; border: none;"></iframe>
  </div>
</div>

<script>
  const toggleBtn = document.getElementById('toggle-chat-btn');
  const frameBox = document.getElementById('chat-frame-box');
  toggleBtn.addEventListener('click', () => {
    frameBox.style.display = frameBox.style.display === 'none' ? 'block' : 'none';
  });
</script>
```
*(Replace `http://localhost:5000` with your deployed backend URL, e.g., on Render or Railway).*

---

## 🚢 Free Cloud Deployment (e.g. Render / Railway)

1. Push this folder to a GitHub repository:
   ```bash
   git init
   git add .
   git commit -m "feat: Arunkumar personal AI chatbot with RAG"
   git branch -M main
   git remote add origin https://github.com/Arunkumar1234KR/personal-ai-chatbot.git
   git push -u origin main
   ```
2. Log in to [Render](https://render.com/) or [Railway](https://railway.app/).
3. Create a **New Web Service**, select your GitHub repository.
4. Set Build Command: `pip install -r requirements.txt`
5. Set Start Command: `gunicorn app:app`
6. Click **Deploy**! Your chatbot backend is now live for free with HTTPS!

---

## 👤 Author & Contact
- **Arunkumar K R**
- **Email**: [krarun2504@gmail.com](mailto:krarun2504@gmail.com)
- **Phone**: +91 8778588180
- **Portfolio**: [https://krarun-portfolio.netlify.app/](https://krarun-portfolio.netlify.app/)
- **LinkedIn**: [linkedin.com/in/arunkumar-kr-3364012b3](https://www.linkedin.com/in/arunkumar-kr-3364012b3)
- **GitHub**: [github.com/Arunkumar1234KR](https://github.com/Arunkumar1234KR)
