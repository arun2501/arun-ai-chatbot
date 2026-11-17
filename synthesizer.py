"""
synthesizer.py
Dual-Engine Response Synthesizer for Arunkumar K R's Personal Chatbot.
Modes:
1. Local Synthesizer: 100% offline, requires ZERO API keys, extracts facts & formats multi-angle first-person answers.
2. Free Cloud LLM (Groq / Gemini / Ollama / OpenRouter): Uses retrieved RAG context when a free key is present in .env.
Seamless fallback: If LLM fails or is unconfigured, automatically generates the high-quality local RAG response.
"""

import os
import sys
import re
import json
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv
from rag_engine import Chunk, clean_and_normalize

# Load environment variables from .env
load_dotenv()

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

try:
    import requests
except ImportError:
    requests = None

# Free LLM Configuration from .env
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").strip()
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2").strip()

SYSTEM_PROMPT = """You are Arunkumar K R (Arun) speaking through your official personal AI portfolio bot.
You are interacting with recruiters, tech leads, interviewers, and professional contacts.

RULES FOR INDUSTRY-GRADE, SMART RESPONSES:
1. Voice & Persona: Always speak authentically in the first person ('I', 'me', 'my') as Arunkumar K R.
2. Direct & Concise: Answer ONLY what was specifically asked. Do NOT dump my entire life history or unrelated projects unless requested.
3. User-Friendly Structure: Always format responses with clear Markdown headings, crisp bullet points, and bold key terms. NEVER output dense, bulk paragraphs.
4. Grounded in Ground Truth: Strictly use the provided RAG context facts:
   - College: Final Year B.Tech in Information Technology (Honors) at Velammal College of Engineering & Technology, Madurai (CGPA: 8.84 / 10).
   - 12th HSC: 92.50% (Maths & Science with CS, 2022, Muthuthevar Mukkulathore Hr. Sec. School).
   - 10th SSLC: 85.40% (2020, Muthuthevar Mukkulathore Hr. Sec. School).
   - Internships: Full Stack Developer Intern at Elysium Technologies (Built Asset Management System with Spring Boot, React, MySQL) & Web Dev Intern at Reccsar.
   - Top Project: Automated Portable Nebulization System (APN) - IoT healthcare device that monitors pulse and automates nebulization (3rd Place National Award in ISTI).
   - Publications: Dual IEEE Xplore author (ML Smart e-PDS + Fruit Health Analysis using MMCNN & Genetic Algorithms).
   - Skills: Java, Spring Boot, React.js, MySQL, Python, Machine Learning, Figma UI/UX.
   - Contact: +91 8778588180, krarun2504@gmail.com, Madurai, open to relocate & remote.
5. Links: Include exact Markdown links where relevant (GitHub repository, live Vercel demo, IEEE Xplore paper, resume PDF).
6. Tone: Highly intelligent, articulate, humble, enthusiastic, and ready for software engineering opportunities.
"""

class ResponseSynthesizer:
    def __init__(self):
        # Refresh from environment
        load_dotenv()
        self.groq_key = os.environ.get("GROQ_API_KEY", "").strip()
        self.gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
        self.openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()

    def update_keys(self, groq_key: str = None, gemini_key: str = None):
        """Allow runtime setting of free API keys from UI or settings."""
        if groq_key is not None:
            self.groq_key = groq_key.strip()
            os.environ["GROQ_API_KEY"] = self.groq_key
        if gemini_key is not None:
            self.gemini_key = gemini_key.strip()
            os.environ["GEMINI_API_KEY"] = self.gemini_key

    def synthesize(self, query: str, retrieved_chunks: List[Tuple[Chunk, float]]) -> Dict[str, Any]:
        """Generate response using Free LLM (if configured) or Local Intelligent Synthesizer."""
        context_text = self._build_context_text(retrieved_chunks)
        suggested_chips = self._generate_suggested_chips(query, retrieved_chunks)

        # 1. Try Free Groq API (LLaMA 3.3 70B / LLaMA 3.1 8B)
        groq_api_key = self.groq_key or os.environ.get("GROQ_API_KEY", "").strip()
        if groq_api_key:
            self.groq_key = groq_api_key
            llm_resp = self._call_groq(query, context_text)
            if llm_resp:
                return {
                    "answer": llm_resp,
                    "engine": "Groq LLaMA 3.3 (Cloud AI)",
                    "sources": [c.title for c, _ in retrieved_chunks[:3]],
                    "suggested_chips": suggested_chips
                }

        # 2. Try Free Google Gemini API (Gemini 2.0 / 1.5 Flash Free Tier)
        if self.gemini_key:
            gemini_resp = self._call_gemini(query, context_text)
            if gemini_resp:
                return {
                    "answer": gemini_resp,
                    "engine": "Google Gemini Flash (Free Cloud LLM)",
                    "sources": [c.title for c, _ in retrieved_chunks[:3]],
                    "suggested_chips": suggested_chips
                }

        # 3. Try Local Ollama (if running)
        ollama_resp = self._call_ollama(query, context_text)
        if ollama_resp:
            return {
                "answer": ollama_resp,
                "engine": f"Local Ollama ({OLLAMA_MODEL})",
                "sources": [c.title for c, _ in retrieved_chunks[:3]],
                "suggested_chips": suggested_chips
            }

        # 4. Built-in Local RAG Synthesizer (Works 100% offline with zero keys!)
        local_resp = self._local_synthesize(query, retrieved_chunks)
        return {
            "answer": local_resp,
            "engine": "Local Neural RAG Synthesizer (Zero API Keys Required)",
            "sources": [c.title for c, _ in retrieved_chunks[:3]],
            "suggested_chips": suggested_chips
        }

    def _build_context_text(self, chunks: List[Tuple[Chunk, float]]) -> str:
        ctx_parts = []
        for i, (chunk, score) in enumerate(chunks):
            ctx_parts.append(f"--- Document [{i+1}] ({chunk.title} | {chunk.category}) ---\n{chunk.content}")
            if chunk.links:
                ctx_parts.append(f"Direct Links: {json.dumps(chunk.links)}")
        return "\n\n".join(ctx_parts)

    def _local_synthesize(self, query: str, retrieved_chunks: List[Tuple[Chunk, float]]) -> str:
        """High quality, comprehensive local response generator when no API key is set."""
        q_clean = clean_and_normalize(query).lower().strip()
        q_lower = query.lower().strip()

        # Handle Greetings with exact word boundaries (avoid matching 'hire', 'him', 'this')
        if re.search(r'\b(hello|hi|hey|greetings|good morning|good afternoon|good evening|vanakkam)\b', q_lower) or "வணக்கம்" in q_lower:
            return (
                "👋 **Hello! Welcome to Arunkumar K R's Personal AI Assistant.**\n\n"
                "I am here to tell you everything about Arun's journey—including his **education at Velammal College (8.84 CGPA)**, "
                "**industry internships at Elysium Technologies & Reccsar**, his **national award-winning IoT Pulse Nebulizer project**, "
                "**2 IEEE research publications**, and his expertise in **Java, Spring Boot, React.js, and Machine Learning**.\n\n"
                "What would you like to explore today? You can ask me anything or click a suggestion below!"
            )

        # Handle Goodbye / Thanks
        if re.search(r'\b(bye|goodbye|thanks|thank you|see you|cya)\b', q_lower) or "நன்றி" in q_lower:
            return (
                "You're very welcome! Feel free to reach out to Arun directly at **krarun2504@gmail.com** or connect on "
                "[LinkedIn](https://www.linkedin.com/in/arunkumar-kr-3364012b3). Have a wonderful day!"
            )

        # Handle Specific 10th / SSLC query (exact, precise answer)
        if any(w in q_clean for w in ["10th", "tenth", "sslc", "secondary school"]):
            return (
                "### 🎓 10th Standard (SSLC) Academic Score\n\n"
                "In his **10th Grade (SSLC - Secondary School Leaving Certificate)** in **2020**, Arunkumar scored:\n\n"
                "• **Academic Score / Percentage**: **85.40%**\n"
                "• **School**: Muthuthevar Mukkulathore Higher Secondary School, Madurai\n"
                "• **Year of Passing**: 2020\n\n"
                "*For further context: Arun later scored 92.50% in his 12th HSC and is currently maintaining an 8.84 CGPA in B.Tech IT (Honors) at Velammal College.*"
            )

        # Handle Specific 12th / HSC query (exact, precise answer)
        if any(w in q_clean for w in ["12th", "twelfth", "hsc", "plus two", "+2", "higher secondary"]):
            return (
                "### 🎓 12th Standard (HSC) Academic Score\n\n"
                "In his **12th Grade (HSC - Higher Secondary Certificate)** in **2022**, Arunkumar scored:\n\n"
                "• **Academic Score / Percentage**: **92.50%**\n"
                "• **Stream**: Mathematics and Science with Computer Science\n"
                "• **School**: Muthuthevar Mukkulathore Higher Secondary School, Madurai\n"
                "• **Year of Passing**: 2022\n\n"
                "*Arun also scored 85.40% in 10th SSLC and holds an 8.84 CGPA in B.Tech IT (Honors).*"
            )

        # Handle College / CGPA / B.Tech
        if any(w in q_clean for w in ["cgpa", "btech", "degree", "velammal", "vcet"]) or ("college" in q_clean and any(w in q_clean for w in ["score", "marks", "grade", "percentage", "cgpa", "gpa"])):
            return (
                "### 🎓 College & CGPA (B.Tech IT Honors)\n\n"
                "Arunkumar is currently in his final year of undergraduate engineering:\n\n"
                "• **Degree**: Bachelor of Technology (B.Tech) in Information Technology (Honors)\n"
                "• **Cumulative GPA (CGPA)**: **8.84 / 10** *(8.76 / 10)*\n"
                "• **College / Institution**: Velammal College of Engineering and Technology, Madurai\n"
                "• **Duration**: 2022 – 2026 (Final Year Pursuing)\n"
                "• **Key Focus**: Full Stack Web Development, Machine Learning, Data Analytics"
            )

        # Handle General Education / Academic Record
        if any(w in q_clean for w in ["education", "academics", "school", "marks", "grades", "percentage", "score"]):
            return (
                "### 🎓 Arunkumar's Complete Academic Record\n\n"
                "Here is Arun's educational background from school to engineering:\n\n"
                "1. **B.Tech in Information Technology (Honors)**: **8.84 CGPA**\n"
                "   * *College*: Velammal College of Engineering and Technology, Madurai (2022 - 2026)\n"
                "   * *Status*: Final Year Pursuing\n\n"
                "2. **Higher Secondary Certificate (12th HSC)**: **92.50%** (2022)\n"
                "   * *Stream*: Maths & Science with Computer Science\n"
                "   * *School*: Muthuthevar Mukkulathore Higher Secondary School\n\n"
                "3. **Secondary School Leaving Certificate (10th SSLC)**: **85.40%** (2020)\n"
                "   * *School*: Muthuthevar Mukkulathore Higher Secondary School"
            )

        # Handle "Why hire Arunkumar?"
        if ("why" in q_clean and ("hire" in q_clean or "choose" in q_clean)) or ("hire" in q_clean and "reason" in q_clean):
            return (
                "### 🌟 Why You Should Hire Arunkumar K R\n\n"
                "Here are the top reasons why Arun is an exceptional candidate for software engineering teams:\n\n"
                "1. **Full-Stack Production Experience**: At **Elysium Technologies**, Arun engineered an enterprise Asset Management System using Spring Boot microservices, Spring Data JPA, React.js, and MySQL. He deployed live solutions on Vercel and Render.\n"
                "2. **Proven Innovation & Awards**: He won **3rd Place in the National-Level Project Contest (ISTI)** for his Pulse-Based Automated Portable Nebulizer (APN).\n"
                "3. **Research-Backed Technical Depth**: He is a **dual IEEE Xplore author**, having developed Multi-Modal CNNs with Genetic Algorithm optimization for agricultural diagnostics and AI crowd-monitoring for public welfare.\n"
                "4. **Strong Academic Foundation**: Maintaining an **8.84 CGPA** in B.Tech IT (Honors) demonstrates dedication, discipline, and quick learning capabilities.\n"
                "5. **Certified & Adaptable**: Holds credentials from **Wipro (Java Full Stack)**, **IBM (AI)**, **Microsoft (Cybersecurity)**, and **TCS iON**."
            )

        # Handle Strengths & Capabilities (supports both singular 'strength' and plural 'strengths')
        if re.search(r'\b(strength|strengths|strong point|strong points|capabilities|good at)\b', q_clean):
            return (
                "### 💪 Arunkumar's Core Strengths & Capabilities\n\n"
                "• **Full-Stack Engineering Depth**: Strong mastery in architecting robust backends with Java, Spring Boot, and MySQL, paired with dynamic React.js frontends.\n"
                "• **Practical Innovation**: Demonstrated ability to turn complex concepts into working systems, winning **3rd Place nationally in the ISTI contest** for the IoT APN Nebulizer.\n"
                "• **Research-Driven Rigor**: Published **two peer-reviewed IEEE papers** applying Multi-Modal CNNs, Genetic Algorithms, and facial recognition AI to real-world challenges.\n"
                "• **High Adaptability & Quick Learning**: Maintained an **8.84 CGPA** in B.Tech IT (Honors) while acquiring multi-domain proficiency across AI/ML, cloud deployment, and UI/UX.\n"
                "• **Professional Reliability**: Proven teamwork, disciplined time management, and commitment to building clean, maintainable software."
            )

        # Handle Weaknesses / Areas of Improvement
        if re.search(r'\b(weakness|weaknesses|areas of improvement|areas to improve|shortcoming|growth areas)\b', q_clean):
            return (
                "### 🎯 Arunkumar's Areas of Improvement & Growth\n\n"
                "• **Deep Detail Orientation**: Arunkumar tends to dive very deep into low-level architectural optimizations. To keep projects agile, he applies strict sprint timeboxing and focuses on delivering working MVPs first.\n"
                "• **Continuous Public Speaking**: While confident in technical presentations (such as presenting at IEEE conferences and project contests), he continuously challenges himself to participate in more open hackathon pitches and community tech talks."
            )

        # Handle Projects Listing / Overview
        if re.search(r'\b(projects|project|portfolio|built|developed|created|showcase|list projects)\b', q_clean) and not any(w in q_clean for w in ["nebulizer", "apn", "epds", "pds", "fruit", "asset", "jewelry", "jewellery"]):
            return (
                "### 🚀 Arunkumar's Top 5 Engineering & Research Projects\n\n"
                "1. **Automated Portable Nebulization System (APN)** | *IoT & Smart Healthcare*\n"
                "   • Real-time pulse-sensing portable nebulizer that detects respiratory distress and triggers automated medication.\n"
                "   • 🏆 **Award**: Secured **3rd Place in the National-Level Project Contest (ISTI)**.\n\n"
                "2. **ML-Based Smart e-PDS** | *AI/ML & Social Good*\n"
                "   • Public Distribution System with facial recognition verification, crowd queue monitoring, and cashless UPI.\n"
                "   • 📜 **Publication**: Published in **IEEE Xplore (Nov 2025)** — [Read Paper](https://ieeexplore.ieee.org/abstract/document/11382690)\n\n"
                "3. **ML-Based Fruit Health & Quality Analysis** | *Computer Vision & Deep Learning*\n"
                "   • Multi-Modal CNN (MMCNN) fusing RGB images with PCA-reduced spectral data, tuned with Genetic Algorithms.\n"
                "   • 📜 **Publication**: Published in **IEEE Xplore (June 2025)** — [Read Paper](https://ieeexplore.ieee.org/abstract/document/11012639)\n\n"
                "4. **Enterprise Asset Management System** | *Full Stack Web Development*\n"
                "   • Production-grade platform built at Elysium Technologies with Spring Boot, Spring Data JPA, React.js, and MySQL.\n"
                "   • 🌐 [Live Demo](https://asset-frontend-xi.vercel.app/) | 💻 [GitHub Source](https://github.com/Arunkumar1234KR/Asset_Management_System)\n\n"
                "5. **Jewelry E-Commerce UX Design** | *UI/UX & Product Design*\n"
                "   • High-fidelity design system, interactive prototypes, and luxury user journeys crafted in Figma."
            )

        # Handle Nebulizer / APN Project
        if re.search(r'\b(nebulizer|apn|pulse nebulizer|asthma|copd)\b', q_clean):
            return (
                "### 🩺 Automated Portable Nebulization System (APN)\n\n"
                "• **Domain**: IoT & Smart Healthcare (Embedded Systems + Sensor Integration)\n"
                "• **Innovation**: Continuously monitors patient pulse rate to detect sudden respiratory distress or asthma attacks in real-time, automatically triggering dynamic nebulization dosages.\n"
                "• **Hardware & Tech**: Microcontrollers, Pulse Sensors, Piezoelectric Mesh Atomizer, C++\n"
                "• **🏆 Award & Recognition**: Secured **3rd Place in the National-Level Project Contest (ISTI)** competing against top engineering institutions.\n"
                "• **Impact**: Provides rapid, automated, life-saving medication delivery for asthma and COPD patients without requiring manual intervention."
            )

        # Handle Smart e-PDS Project
        if re.search(r'\b(epds|pds|ration|smart epds|fair price)\b', q_clean):
            return (
                "### 🌾 ML-Based Smart e-Public Distribution System (e-PDS)\n\n"
                "• **Domain**: Computer Vision, Deep Learning & Public Welfare Automation\n"
                "• **Innovation**: Eliminates ration leakage and ghost beneficiaries using real-time facial recognition verification, automated queue crowd-density monitoring, and secure cashless UPI payments.\n"
                "• **Tech Stack**: Python, OpenCV, CNN Facial Recognition, Flask, Cashless Payment Gateway\n"
                "• **📜 IEEE Publication**: Published in **IEEE Xplore (Nov 2025)**\n"
                "• 🔗 **[Read Paper on IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/11382690)**"
            )

        # Handle Fruit Health Analysis Project
        if re.search(r'\b(fruit|fruits|mmcnn|spectral|grading|citrus)\b', q_clean):
            return (
                "### 🍎 ML-Based Fruit Health & Quality Analysis System\n\n"
                "• **Domain**: Multi-Modal Deep Learning & Precision Agriculture\n"
                "• **Innovation**: Fuses standard RGB imagery with PCA-reduced spectral sensor data via a Multi-Modal CNN (MMCNN), optimized using Genetic Algorithms for high-precision grading.\n"
                "• **Tech Stack**: Python, Multi-Modal CNN (MMCNN), PCA, Genetic Algorithms, OpenCV\n"
                "• **📜 IEEE Publication**: Published in **IEEE Xplore (June 2025)**\n"
                "• 🔗 **[Read Paper on IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/11012639)**"
            )

        # Handle Asset Management System Project
        if re.search(r'\b(asset|asset management|elysium project|inventory)\b', q_clean):
            return (
                "### 💼 Enterprise Asset Management System\n\n"
                "• **Domain**: Enterprise Full Stack Web Application (Engineered at Elysium Technologies)\n"
                "• **Core Features**: Asset lifecycle tracking, maintenance scheduling, vendor management, and role-based access control (RBAC).\n"
                "• **Tech Stack**: Java, Spring Boot, Spring Data JPA, React.js, MySQL, Tailwind CSS\n"
                "• **Deployment**: Frontend hosted on **Vercel**, backend microservices on **Render**.\n"
                "• 🌐 **[Live Demo](https://asset-frontend-xi.vercel.app/)** | 💻 **[GitHub Source](https://github.com/Arunkumar1234KR/Asset_Management_System)**"
            )

        # Handle Jewelry UX/UI Project
        if re.search(r'\b(jewelry|jewellery|figma|ux|ui|wireframe|prototype)\b', q_clean) and not ("skill" in q_clean):
            return (
                "### 💎 Jewelry E-Commerce UX/UI Design\n\n"
                "• **Domain**: Product Design & UI/UX\n"
                "• **Focus**: Created an intuitive, luxury mobile and web shopping experience for fine jewelry.\n"
                "• **Toolkit**: Figma, Interactive Prototyping, User Flow Mapping, Design Systems\n"
                "• **Deliverables**: Complete design component library, interactive micro-interactions, and high-converting checkout flows."
            )

        # Handle Internships & Work Experience Overview
        if re.search(r'\b(internship|internships|experience|work experience|companies|where did you work)\b', q_clean) and not any(w in q_clean for w in ["elysium", "reccsar"]):
            return (
                "### 💼 Arunkumar's Industry Internships & Experience\n\n"
                "Arunkumar has completed two professional software engineering internships:\n\n"
                "1. **Full Stack Developer Intern — Elysium Technologies, Madurai** *(Feb 2024 – Mar 2024)*\n"
                "   • Engineered the enterprise **Asset Management System** using Spring Boot, React.js, and MySQL.\n"
                "   • Built secure REST APIs with Spring Data JPA and deployed client on Vercel and backend on Render.\n"
                "   • 🌐 [Live App](https://asset-frontend-xi.vercel.app/) | 💻 [Source Code](https://github.com/Arunkumar1234KR/Asset_Management_System)\n\n"
                "2. **Web Development Intern — Reccsar Private Limited, Madurai** *(Jun 2023 – Jul 2023)*\n"
                "   • Built responsive e-commerce web modules using HTML5, CSS3, JavaScript, and PHP/MySQL.\n"
                "   • Optimized client-side page load speed, cross-device responsiveness, and database query latency."
            )

        # Handle Elysium Technologies Internship
        if "elysium" in q_clean:
            return (
                "### 🏢 Full Stack Developer Internship at Elysium Technologies\n\n"
                "• **Role**: Full Stack Developer Intern *(Feb 2024 – Mar 2024 | Madurai, India)*\n"
                "• **Core Product**: Built the enterprise **Asset Management System** from ground up.\n"
                "• **Key Responsibilities & Deliverables**:\n"
                "  - Architected backend RESTful microservices with **Java & Spring Boot**.\n"
                "  - Modeled relational schemas and optimized queries using **MySQL** and **Spring Data JPA**.\n"
                "  - Developed dynamic, component-driven user interfaces using **React.js** and **Tailwind CSS**.\n"
                "  - Deployed the application live: frontend on **Vercel** and backend APIs on **Render**.\n\n"
                "🌐 **[Live Demo](https://asset-frontend-xi.vercel.app/)** | 💻 **[GitHub Repository](https://github.com/Arunkumar1234KR/Asset_Management_System)**"
            )

        # Handle Reccsar Internship
        if "reccsar" in q_clean:
            return (
                "### 🏢 Web Development Internship at Reccsar Private Limited\n\n"
                "• **Role**: Web Development Intern *(Jun 2023 – Jul 2023 | Madurai, India)*\n"
                "• **Core Focus**: E-Commerce web platform modules and interface engineering.\n"
                "• **Key Responsibilities & Deliverables**:\n"
                "  - Built responsive, interactive web components using **HTML5, CSS3, and JavaScript**.\n"
                "  - Integrated client interfaces with backend **PHP & MySQL** services.\n"
                "  - Enhanced cross-browser compatibility, mobile touch responsiveness, and page loading speed."
            )

        # Handle Technical Skills & Tech Stack
        if re.search(r'\b(skill|skills|tech stack|technologies|tools|languages|programming)\b', q_clean) and not any(w in q_clean for w in ["java", "python", "react", "spring", "mysql", "figma"]):
            return (
                "### 🛠️ Arunkumar's Technical Skills Toolkit\n\n"
                "• **Programming Languages**: Java (Core & OOP), Python, JavaScript (ES6+), C\n"
                "• **Backend Frameworks**: Spring Boot, Spring Data JPA, RESTful APIs, Microservices, Flask\n"
                "• **Frontend Technologies**: React.js, HTML5, CSS3, JavaScript, Tailwind CSS, Bootstrap\n"
                "• **Databases & Cloud**: MySQL, Firebase (Firestore/Auth), Vercel, Render\n"
                "• **AI & Machine Learning**: CNN, Multi-Modal CNN (MMCNN), Genetic Algorithms, PCA, Face Recognition, RAG Pipelines\n"
                "• **UI/UX & Development Tools**: Figma, Git, GitHub, Postman, IoT Microcontrollers"
            )

        # Handle Java & Spring Boot
        if re.search(r'\b(java|spring|spring boot)\b', q_clean):
            return (
                "### ☕ Java & Spring Boot Expertise\n\n"
                "• **Core Java**: Solid foundation in Object-Oriented Programming (OOP), Collections Framework, Multithreading, and Exception Handling.\n"
                "• **Spring Boot**: Experienced building production-ready RESTful microservices, Spring Data JPA repositories, validation, and Maven build automation.\n"
                "• **Industry Validation**: Developed the enterprise Asset Management System at Elysium Technologies and holds the **Wipro TalentNext Java Full Stack certification**."
            )

        # Handle Python & AI/ML
        if re.search(r'\b(python|machine learning|deep learning|data analysis)\b', q_clean) and not any(w in q_clean for w in ["fruit", "epds"]):
            return (
                "### 🐍 Python & Machine Learning Expertise\n\n"
                "• **Data Science & ML Libraries**: NumPy, Pandas, Scikit-Learn, OpenCV, Matplotlib, Flask, TensorFlow/Keras.\n"
                "• **Applied AI Systems**: Multi-Modal CNN (MMCNN) fusing RGB and spectral data, Genetic Algorithm optimization, and real-time facial recognition.\n"
                "• **IEEE Publications**: Authored two peer-reviewed IEEE papers powered by custom Python machine learning architectures."
            )

        # Handle React.js & Frontend
        if re.search(r'\b(react|reactjs|frontend|html|css|javascript)\b', q_clean) and not any(w in q_clean for w in ["reccsar"]):
            return (
                "### ⚛️ Frontend & React.js Expertise\n\n"
                "• **React.js**: Functional components, React Hooks (`useState`, `useEffect`, `useContext`), state management, React Router, and Axios.\n"
                "• **Web Standards**: Semantic HTML5, modern CSS3, ES6+ JavaScript, responsive Flexbox and Grid layouts.\n"
                "• **Styling**: Tailwind CSS and Bootstrap 5 for sleek, modern, mobile-first web interfaces.\n"
                "• **Live Project**: Built the frontend for Elysium's Asset Management System — [Live Demo](https://asset-frontend-xi.vercel.app/)."
            )

        # Handle Research Papers & IEEE Publications
        if re.search(r'\b(paper|papers|publication|publications|research|ieee|xplore)\b', q_clean):
            return (
                "### 📜 Arunkumar's Published IEEE Research Papers\n\n"
                "Arunkumar has authored **two peer-reviewed research papers** published in **IEEE Xplore**:\n\n"
                "1. **ML based Smart e - Public Distribution System** *(Published: Nov 2025)*\n"
                "   • Features automated facial recognition verification, crowd queue monitoring, and cashless UPI payments.\n"
                "   • 🔗 **[Read on IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/11382690)**\n\n"
                "2. **ML Based Fruit Health and Quality Analysis System** *(Published: June 2025)*\n"
                "   • Multi-Modal CNN fusing RGB imagery with PCA-reduced spectral data, tuned via Genetic Algorithms.\n"
                "   • 🔗 **[Read on IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/11012639)**"
            )

        # Handle Industry Certifications
        if re.search(r'\b(certification|certifications|certificate|certificates|credentials|courses|licensed)\b', q_clean):
            return (
                "### 📜 Industry Certifications & Credentials\n\n"
                "• **Wipro TalentNext — Java Full Stack** *(Digital Skills Readiness Program)*: Enterprise Java, Spring Boot, Microservices, React.\n"
                "• **IBM — Artificial Intelligence**: Machine learning principles, neural networks, and AI fundamentals.\n"
                "• **TCS iON — Career Edge (Young Professional)**: Corporate readiness, agile teamwork, and business communication.\n"
                "• **Microsoft — Cybersecurity Fundamentals**: Network security, identity management, and threat mitigation.\n"
                "• **NPTEL — Academic Certifications**: Subject credentials in computer science and programming fundamentals."
            )

        # Handle Honors & Awards
        if re.search(r'\b(award|awards|achievement|achievements|honor|honors|contest|competitions|prize|won)\b', q_clean):
            return (
                "### 🏆 Honors & Competitive Achievements\n\n"
                "• **3rd Place — National-Level Project Contest (ISTI)**: Won 3rd place nationally for the Automated Portable Nebulization System (APN).\n"
                "• **Dual IEEE Author**: Published two international research papers in IEEE Xplore as an undergraduate researcher.\n"
                "• **TCE Technical Paper Commendation**: Commended for technical paper presentation at Thiagarajar College of Engineering.\n"
                "• **Consistent Academic Merit**: Maintained an **8.84 CGPA** in B.Tech IT (Honors) at Velammal College."
            )

        # Handle Location & Relocation
        if re.search(r'\b(location|live|where are you|where do you live|address|city|native|thirunagar|madurai)\b', q_clean):
            return (
                "### 📍 Location & Relocation Preferences\n\n"
                "• **Current Location**: 54 Srinivasa Nagar, Thirunagar, Madurai - 625006, Tamil Nadu, India\n"
                "• **Hometown**: Madurai, Tamil Nadu\n"
                "• **Relocation Status**: Open and enthusiastic about relocating to major tech hubs (Bengaluru, Chennai, Hyderabad, Pune, Mumbai, etc.)\n"
                "• **Work Mode**: Open to On-Site, Hybrid, and Remote opportunities."
            )

        # Handle Phone / WhatsApp
        if re.search(r'\b(phone|mobile|call|whatsapp|cell|number)\b', q_clean) and not any(w in q_clean for w in ["cgpa", "percentage", "marks"]):
            return (
                "### 📱 Phone & Direct Contact\n\n"
                "• **Mobile / Call**: [+91 8778588180](tel:+918778588180)\n"
                "• **WhatsApp**: [Chat on WhatsApp (+91 8778588180)](https://wa.me/918778588180)\n"
                "• **Location**: Madurai, Tamil Nadu, India"
            )

        # Handle Email
        if re.search(r'\b(email|mail)\b', q_clean) and not any(w in q_clean for w in ["phone", "call"]):
            return (
                "### 📧 Email Address\n\n"
                "• **Primary Email**: [krarun2504@gmail.com](mailto:krarun2504@gmail.com)\n"
                "• **LinkedIn**: [linkedin.com/in/arunkumar-kr-3364012b3](https://www.linkedin.com/in/arunkumar-kr-3364012b3)"
            )

        # Handle Birthday / Age
        if re.search(r'\b(birthday|birth date|dob|date of birth|age)\b', q_clean):
            return (
                "### 🎂 Birthday & Personal Details\n\n"
                "• **Date of Birth**: 25 November 2004\n"
                "• **Languages**: English (Professional Working Proficiency), Tamil (Native)\n"
                "• **Location**: Madurai, Tamil Nadu, India"
            )

        # Handle Languages
        if re.search(r'\b(language|languages|speak|tamil|english)\b', q_clean) and not any(w in q_clean for w in ["programming", "code", "coding", "java", "python"]):
            return (
                "### 🗣️ Languages Known\n\n"
                "• **English**: Full professional proficiency for technical documentation, engineering teamwork, and client interactions.\n"
                "• **Tamil**: Native proficiency."
            )

        # Handle Persona / Introduction / Bio
        if re.search(r'\b(who are you|tell me about yourself|introduce|who is arun|about you|bio)\b', q_clean):
            return (
                "### 👋 About Arunkumar K R\n\n"
                "I am **Arunkumar K R**, a passionate **Full Stack Developer, Data Analyst, and AI/ML Engineer** pursuing **B.Tech in Information Technology (Honors)** at Velammal College of Engineering and Technology, Madurai (CGPA: **8.84 / 10**).\n\n"
                "• **Core Focus**: Enterprise web applications (Spring Boot + React.js), IoT healthcare systems, and Computer Vision.\n"
                "• **Industry Experience**: Completed software engineering internships at **Elysium Technologies** and **Reccsar**.\n"
                "• **Research & Awards**: Dual **IEEE published author** and **3rd place winner** in a national-level IoT innovation contest.\n\n"
                "Feel free to ask me about my projects, skills, education, or how to get in touch!"
            )

        # Handle Direct Contact / Phone / Email / Hire
        if any(w in q_clean for w in ["contact", "phone", "email", "number", "reach", "call", "whatsapp", "hire him", "how to hire", "get in touch"]):
            return (
                "### 📞 How to Contact & Hire Arunkumar K R\n\n"
                "Arun is actively open to **Full-Time Software Engineering roles, Full Stack/Backend positions, and Internships**!\n\n"
                "• **Phone / WhatsApp**: [+91 8778588180](tel:+918778588180)\n"
                "• **Email Address**: [krarun2504@gmail.com](mailto:krarun2504@gmail.com)\n"
                "• **Current Location**: 54 Srinivasa Nagar, Thirunagar, Madurai - 625006, Tamil Nadu, India *(Open to relocate / remote)*\n"
                "• **LinkedIn**: [linkedin.com/in/arunkumar-kr-3364012b3](https://www.linkedin.com/in/arunkumar-kr-3364012b3)\n"
                "• **GitHub**: [github.com/Arunkumar1234KR](https://github.com/Arunkumar1234KR)\n"
                "• **Portfolio**: [krarun-portfolio.netlify.app](https://krarun-portfolio.netlify.app/)\n\n"
                "📄 **[Download Official CV / Resume (PDF)](https://krarun-portfolio.netlify.app/assets/img/913122205009_KR%20ARUNKUMAR.pdf)**"
            )

        # Handle Resume Download query
        if any(w in q_clean for w in ["resume", "cv", "download resume", "download cv"]):
            return (
                "### 📄 Arunkumar's Official Resume / CV\n\n"
                "You can view and download Arun's latest resume directly here:\n\n"
                "👉 **[Download KR Arunkumar CV (PDF)](https://krarun-portfolio.netlify.app/assets/img/913122205009_KR%20ARUNKUMAR.pdf)**\n\n"
                "Summary highlights from his resume:\n"
                "• **B.Tech Information Technology (Honors)** at Velammal College of Engineering & Technology (CGPA: 8.84)\n"
                "• **Full Stack & ML Developer**: Java, Spring Boot, React.js, MySQL, Python\n"
                "• **Dual IEEE Author** & **3rd Place ISTI National Contest Winner**\n"
                "• **Work Experience**: Elysium Technologies (Asset Management System) & Reccsar (E-Commerce)"
            )

        # Smart, user-friendly fallback from retrieved chunks (never dumps raw web text)
        primary_chunk = retrieved_chunks[0][0]
        for c, s in retrieved_chunks:
            if c.category != "live_web":
                primary_chunk = c
                break

        # Clean content into user-friendly bullet points
        raw_text = primary_chunk.content
        clean_lines = []
        for line in raw_text.split('\n'):
            line = line.strip()
            if not line or line.startswith("Source:") or line.startswith("Content Summary:"):
                continue
            if line.startswith("Question:") or line.startswith("Answer:"):
                clean_lines.append(line.replace("Question:", "").replace("Answer:", "").strip())
            else:
                clean_lines.append(line)

        # Build clean response
        response_lines = [f"### {primary_chunk.title}\n"]
        for cl in clean_lines[:6]:
            if not cl.startswith("•") and not cl.startswith("-") and not re.match(r'^\d+\.', cl):
                response_lines.append(f"• {cl}")
            else:
                response_lines.append(cl)

        # Collect direct links
        all_links = {}
        for c, _ in retrieved_chunks[:3]:
            if c.links and c.category != "live_web":
                all_links.update(c.links)

        if all_links:
            response_lines.append("\n**🔗 Relevant Links:**")
            for label, url in all_links.items():
                clean_label = label.replace('_', ' ').title()
                response_lines.append(f"• [{clean_label}]({url})")

        return "\n".join(response_lines)

    def _generate_suggested_chips(self, query: str, retrieved_chunks: List[Tuple[Chunk, float]]) -> List[str]:
        """Suggest smart follow-up question chips based on query and results."""
        q = query.lower()
        if "project" in q or "nebulizer" in q or "epds" in q or "fruit" in q:
            return ["Tell me about your IEEE publications", "What did you do at Elysium Technologies?", "What are your core technical skills?", "Download Resume"]
        elif "experience" in q or "internship" in q or "elysium" in q or "reccsar" in q:
            return ["Tell me about your top projects", "What is your CGPA & College?", "Why should we hire you?", "How to contact Arunkumar?"]
        elif "skill" in q or "java" in q or "react" in q or "python" in q:
            return ["Tell me about the Asset Management project", "What certifications do you have?", "Are you open to relocation?", "Download CV"]
        elif "contact" in q or "hire" in q or "email" in q or "phone" in q:
            return ["Download Resume (PDF)", "Why should we hire Arunkumar?", "Explore his projects", "View LinkedIn profile"]
        else:
            return ["🚀 Top Projects", "💼 Work Experience", "🎓 Education & CGPA", "📜 IEEE Papers", "📞 Contact & Hire"]

    def _call_groq(self, query: str, context: str) -> str:
        """Call Free Groq Cloud API with verified active models."""
        if not requests or not self.groq_key:
            return None
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.groq_key}",
            "Content-Type": "application/json"
        }
        
        # Priority list of high-speed active models on Groq
        models_to_try = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]
        
        for model in models_to_try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context Information from Arunkumar's Official Records:\n{context}\n\nUser Question: {query}"}
                ],
                "temperature": 0.2,
                "max_tokens": 800
            }
            try:
                res = requests.post(url, headers=headers, json=payload, timeout=8)
                if res.status_code == 200:
                    data = res.json()
                    msg = data["choices"][0]["message"]
                    content = msg.get("content", "").strip()
                    if content:
                        return content
                else:
                    print(f"[Synthesizer] Groq model '{model}' returned status {res.status_code}: {res.text}")
            except Exception as e:
                print(f"[Synthesizer] Groq API error on model '{model}': {e}")
        return None

    def _call_gemini(self, query: str, context: str) -> str:
        """Call Free Google Gemini API (Gemini 2.0 / 1.5 Flash Free Tier)."""
        if not requests or not self.gemini_key:
            return None
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_key}"
        headers = {"Content-Type": "application/json"}
        prompt = f"{SYSTEM_PROMPT}\n\nContext Information from Arunkumar's Records:\n{context}\n\nUser Question: {query}"
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 1024
            }
        }
        try:
            res = requests.post(url, headers=headers, json=payload, timeout=8)
            if res.status_code == 200:
                data = res.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            else:
                print(f"[Synthesizer] Gemini API returned status {res.status_code}: {res.text}")
        except Exception as e:
            print(f"[Synthesizer] Gemini API error: {e}")
        return None

    def _call_ollama(self, query: str, context: str) -> str:
        """Call Local Ollama instance if explicitly enabled."""
        if not requests or os.environ.get("OLLAMA_ENABLED", "").lower() not in ["true", "1"]:
            return None
        url = f"{OLLAMA_URL.rstrip('/')}/api/generate"
        prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nUser Question: {query}"
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
        try:
            res = requests.post(url, json=payload, timeout=4)
            if res.status_code == 200:
                data = res.json()
                return data.get("response", "")
        except Exception:
            pass
        return None

if __name__ == "__main__":
    from rag_engine import RAGEngine
    rag = RAGEngine()
    synth = ResponseSynthesizer()

    sample_queries = [
        "What is Arun's CGPA?",
        "Tell me about the Nebulizer project",
        "Why should our company hire Arunkumar?",
        "How can I contact him?"
    ]
    for q in sample_queries:
        print(f"\n==================== Q: {q} ====================")
        chunks = rag.retrieve(q, top_k=3)
        res = synth.synthesize(q, chunks)
        print(f"Engine: {res['engine']}")
        print(f"Answer:\n{res['answer']}")
        print(f"Suggested chips: {res['suggested_chips']}")
