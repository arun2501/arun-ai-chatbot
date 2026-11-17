"""
rag_engine.py
Production-grade Retrieval-Augmented Generation (RAG) Engine for Arunkumar K R.
Features:
- Multi-source knowledge ingestion (Profile, Education, Skills, Experience, Projects, Publications, Certifications, FAQs, Live Netlify Portfolio)
- Semantic chunking with rich metadata, direct links, and keyword indexing
- Hybrid BM25 / Sublinear TF-IDF retrieval + Exact Entity & N-gram matching
- Domain-specific intent & synonym expansion (e.g. "grades" -> "cgpa", "contact" -> "phone/email")
- Dynamic re-indexing for real-time updates without restarting the application
"""

import os
import json
import math
import re
import difflib
from typing import List, Dict, Any, Tuple

TYPO_MAP = {
    "perrentage": "percentage",
    "persentage": "percentage",
    "precentage": "percentage",
    "percentge": "percentage",
    "percntage": "percentage",
    "percentag": "percentage",
    "percen": "percentage",
    "prcentage": "percentage",
    "mw": "me",
    "tehn": "then",
    "tenth": "10th",
    "twelfth": "12th",
    "colg": "college",
    "clg": "college",
    "colleg": "college",
    "cgpaa": "cgpa",
    "experince": "experience",
    "experiance": "experience",
    "intrenship": "internship",
    "intership": "internship",
    "projct": "project",
    "projets": "project",
    "nebulzer": "nebulizer",
    "nebulisr": "nebulizer",
    "recsar": "reccsar",
    "elysiumm": "elysium",
    "resum": "resume",
    "strenth": "strength",
    "stregth": "strength",
    "strengh": "strength",
    "strenght": "strength",
    "weaknes": "weakness",
    "skil": "skills",
    "skils": "skills",
    "locaton": "location",
    "cert": "certification",
    "certif": "certification",
    "proj": "project"
}

KNOWN_TERMS = [
    "percentage", "cgpa", "college", "school", "education", "internship", "experience",
    "nebulizer", "projects", "publications", "certifications", "contact", "resume",
    "velammal", "elysium", "reccsar", "skills", "python", "spring", "react", "mysql",
    "10th", "12th", "sslc", "hsc", "strength", "weakness", "location"
]

def clean_and_normalize(text: str) -> str:
    """Normalize common typos, handle '10 th' / '12 th', and apply fuzzy correction."""
    q = re.sub(r'\b10\s*th\b', '10th', text.lower())
    q = re.sub(r'\b12\s*th\b', '12th', q)
    words = q.split()
    corrected = []
    for w in words:
        clean_w = re.sub(r'[^a-zA-Z0-9]', '', w)
        if clean_w in TYPO_MAP:
            corrected.append(TYPO_MAP[clean_w])
        else:
            matches = difflib.get_close_matches(clean_w, KNOWN_TERMS, n=1, cutoff=0.78)
            if matches:
                corrected.append(matches[0])
            else:
                corrected.append(w)
    return " ".join(corrected)

class Chunk:
    def __init__(self, chunk_id: str, title: str, category: str, content: str, keywords: List[str] = None, links: Dict[str, str] = None, metadata: Dict[str, Any] = None):
        self.chunk_id = chunk_id
        self.title = title
        self.category = category  # profile, education, experience, project, publication, skill, certification, faq, live_web
        self.content = content.strip()
        self.keywords = [k.lower() for k in (keywords or [])]
        self.links = links or {}
        self.metadata = metadata or {}
        self.tokens: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.chunk_id,
            "title": self.title,
            "category": self.category,
            "content": self.content,
            "keywords": self.keywords,
            "links": self.links,
            "metadata": self.metadata
        }

class RAGEngine:
    def __init__(self, knowledge_dir: str = None):
        if not knowledge_dir:
            knowledge_dir = os.path.join(os.path.dirname(__file__), "knowledge")
        self.knowledge_dir = knowledge_dir
        self.chunks: List[Chunk] = []
        self.inverted_index: Dict[str, List[Tuple[int, float]]] = {}  # token -> [(chunk_idx, tf)]
        self.idf: Dict[str, float] = {}
        self.chunk_lengths: List[int] = []
        self.avg_chunk_length: float = 0.0
        self.total_docs: int = 0
        self.index_knowledge()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Normalize text and extract unigrams + bigrams for robust phrase matching."""
        cleaned = re.sub(r'[^a-zA-Z0-9\+\#\.\s]', ' ', text.lower())
        words = [w for w in cleaned.split() if len(w) > 1 and not w in {"the", "is", "at", "which", "on", "a", "an", "and", "or", "to", "in", "of", "for", "with"}]
        # Generate unigrams
        tokens = list(words)
        # Generate bigrams for technical phrases (e.g., "spring boot", "machine learning", "react js")
        for i in range(len(words) - 1):
            tokens.append(f"{words[i]}_{words[i+1]}")
        return tokens

    def index_knowledge(self):
        """Load all structured knowledge sources and build search indices."""
        self.chunks = []
        if not os.path.exists(self.knowledge_dir):
            os.makedirs(self.knowledge_dir, exist_ok=True)

        self._load_profile()
        self._load_education()
        self._load_skills()
        self._load_experience()
        self._load_projects()
        self._load_publications()
        self._load_certifications()
        self._load_faq()
        self._load_live_portfolio()

        # Build TF-IDF / BM25 index
        self.total_docs = len(self.chunks)
        self.chunk_lengths = []
        doc_term_freqs: List[Dict[str, int]] = []
        df_counts: Dict[str, int] = {}

        for chunk in self.chunks:
            full_text = f"{chunk.title} {chunk.category} {' '.join(chunk.keywords)} {chunk.content}"
            tokens = self.tokenize(full_text)
            chunk.tokens = tokens
            length = len(tokens)
            self.chunk_lengths.append(length)

            freqs: Dict[str, int] = {}
            for t in tokens:
                freqs[t] = freqs.get(t, 0) + 1
            doc_term_freqs.append(freqs)

            for t in set(tokens):
                df_counts[t] = df_counts.get(t, 0) + 1

        self.avg_chunk_length = sum(self.chunk_lengths) / max(self.total_docs, 1)

        # Calculate BM25 IDF
        self.idf = {}
        for term, df in df_counts.items():
            # BM25 standard IDF with smoothing
            self.idf[term] = math.log(1.0 + (self.total_docs - df + 0.5) / (df + 0.5))

        # Build Inverted Index
        self.inverted_index = {}
        for idx, freqs in enumerate(doc_term_freqs):
            doc_len = self.chunk_lengths[idx]
            for term, tf in freqs.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                # BM25 Term Weight
                k1 = 1.2
                b = 0.75
                bm25_tf = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / max(self.avg_chunk_length, 1))))
                self.inverted_index[term].append((idx, bm25_tf))

        print(f"[RAGEngine] Successfully indexed {self.total_docs} knowledge chunks across all domains.")

    def _load_profile(self):
        path = os.path.join(self.knowledge_dir, "profile.json")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Chunk 1: General Bio & Persona
        self.chunks.append(Chunk(
            chunk_id="profile_bio",
            title="Profile & Executive Summary",
            category="profile",
            content=f"Full Name: {data.get('full_name')} (Known as {data.get('preferred_name')}).\n"
                    f"Title: {data.get('title')}.\n"
                    f"Location: {data.get('location')}.\n"
                    f"Objective: {data.get('professional_objective')}\n"
                    f"Overview: {data.get('about_summary')}\n"
                    f"Languages: {', '.join(data.get('languages', []))}.",
            keywords=["who is arun", "about arun", "tell me about yourself", "summary", "overview", "bio", "profile", "languages", "tamil", "english", "location", "madurai"],
            links={"portfolio": data.get("website", ""), "linkedin": data.get("linkedin", ""), "github": data.get("github", "")},
            metadata=data
        ))

        # Chunk 2: Contact, Links & Resume
        self.chunks.append(Chunk(
            chunk_id="profile_contact",
            title="Contact Information, Social Links & Resume Download",
            category="contact",
            content=f"Contact Arunkumar K R:\n"
                    f"• Phone / Mobile: {data.get('phone')}\n"
                    f"• Email Address: {data.get('email')}\n"
                    f"• Postal Address: {data.get('location')}\n"
                    f"• Official Portfolio Website: {data.get('website')}\n"
                    f"• LinkedIn Profile: {data.get('linkedin')}\n"
                    f"• GitHub Repositories: {data.get('github')}\n"
                    f"• Direct Resume Download: {data.get('resume_download_url')}\n"
                    f"Availability: {data.get('work_authorization')}",
            keywords=["contact", "phone", "mobile", "number", "email", "address", "reach", "hire", "call", "whatsapp", "linkedin", "github", "portfolio", "resume", "cv", "download", "link"],
            links={
                "resume": data.get("resume_download_url", ""),
                "portfolio": data.get("website", ""),
                "linkedin": data.get("linkedin", ""),
                "github": data.get("github", "")
            },
            metadata={"phone": data.get("phone"), "email": data.get("email")}
        ))

        # Chunk 3: Soft Skills & Strengths
        self.chunks.append(Chunk(
            chunk_id="profile_strengths",
            title="Strengths, Work Ethic & Soft Skills",
            category="profile",
            content="Arunkumar's Core Strengths & Capabilities:\n" +
                    "\n".join([f"• {s}" for s in data.get("strengths", [])]) +
                    "\n\nKey Soft Skills:\n" +
                    "\n".join([f"• {s}" for s in data.get("soft_skills", [])]),
            keywords=["strengths", "soft skills", "leadership", "teamwork", "communication", "why hire", "work ethic", "problem solving"],
            links={"linkedin": data.get("linkedin", "")}
        ))

    def _load_education(self):
        path = os.path.join(self.knowledge_dir, "education.json")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for edu in data.get("education", []):
            title = edu.get("degree", "")
            if "SSLC" in title or "10th" in title:
                cid = "edu_10th_sslc"
                kw = ["10th", "tenth", "sslc", "secondary", "10", "percentage", "marks", "score", "school", "muthuthevar", "mukkulathore", "2020", "85.40", "85.4%"]
                content = (
                    f"Degree / Exam: 10th Standard (SSLC - Secondary School Leaving Certificate)\n"
                    f"Institution: {edu.get('institution')}, {edu.get('location')}\n"
                    f"Year of Passing: {edu.get('year')}\n"
                    f"Score / Percentage: {edu.get('percentage')} (85.40%)\n"
                    f"Summary: In 2020, Arunkumar scored 85.40% in his 10th Grade SSLC board exams at Muthuthevar Mukkulathore Hr. Sec. School."
                )
            elif "HSC" in title or "12th" in title:
                cid = "edu_12th_hsc"
                kw = ["12th", "twelfth", "hsc", "plus two", "+2", "higher secondary", "12", "percentage", "marks", "score", "school", "muthuthevar", "mukkulathore", "2022", "92.50", "92.5%"]
                content = (
                    f"Degree / Exam: 12th Standard (HSC - Higher Secondary Certificate)\n"
                    f"Institution: {edu.get('institution')}, {edu.get('location')}\n"
                    f"Year / Duration: {edu.get('duration', edu.get('year'))}\n"
                    f"Score / Percentage: {edu.get('percentage')} (92.50%)\n"
                    f"Stream: {edu.get('stream', 'Mathematics and Science with Computer Science')}\n"
                    f"Summary: In 2022, Arunkumar scored an outstanding 92.50% in his 12th Grade HSC board exams at Muthuthevar Mukkulathore Hr. Sec. School."
                )
            else:
                cid = "edu_college"
                kw = ["college", "btech", "degree", "cgpa", "engineering", "velammal", "vcet", "honors", "grades", "score", "8.84", "8.76", "it", "information technology"]
                content = (
                    f"Degree: {title}\n"
                    f"Institution: {edu.get('institution')}, {edu.get('location')}\n"
                    f"Duration: {edu.get('duration')}\n"
                    f"CGPA: {edu.get('cgpa')}\n"
                    f"Status: {edu.get('status')}\n"
                    f"Highlights: {edu.get('highlights')}\n"
                    f"Relevant Coursework: {', '.join(edu.get('key_courses', []))}"
                )

            self.chunks.append(Chunk(
                chunk_id=cid,
                title=f"Education: {title}",
                category="education",
                content=content,
                keywords=kw,
                metadata=edu
            ))

        # Consolidated education overview
        self.chunks.append(Chunk(
            chunk_id="edu_overview",
            title="Complete Academic Record & Education Timeline",
            category="education",
            content=(
                "Arunkumar's Complete Educational History:\n"
                "1. College (B.Tech IT Honors): Velammal College of Engineering and Technology (2022 - 2026), CGPA: 8.84 / 10.\n"
                "2. 12th Standard (HSC): Muthuthevar Mukkulathore Hr. Sec. School (2020 - 2022), Score: 92.50% (Maths & Science with CS).\n"
                "3. 10th Standard (SSLC): Muthuthevar Mukkulathore Hr. Sec. School (2020), Score: 85.40%."
            ),
            keywords=["education", "school", "college", "cgpa", "percentage", "marks", "scores", "academics", "timeline"]
        ))

    def _load_skills(self):
        path = os.path.join(self.knowledge_dir, "skills.json")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for cat in data.get("categories", []):
            cat_name = cat.get("name")
            skills_list = cat.get("skills", [])
            lines = [f"Category: {cat_name}"]
            kw = ["skills", "tech stack", "technologies", cat_name.lower()]
            for s in skills_list:
                lines.append(f"• {s.get('name')} ({s.get('level')}): {s.get('description')}")
                kw.append(s.get('name').lower())

            self.chunks.append(Chunk(
                chunk_id=f"skill_{cat_name.lower().replace(' ', '_')}",
                title=f"Technical Skills: {cat_name}",
                category="skill",
                content="\n".join(lines),
                keywords=kw,
                metadata={"category": cat_name, "skills": skills_list}
            ))

        # Consolidated skills overview
        all_skills_lines = [
            "Arunkumar's Technical Skills Toolkit & Core Competencies:",
            "• Programming Languages: Java (Core, OOP, Collections), Python, JavaScript (ES6+), C",
            "• Backend & Frameworks: Spring Boot, Spring Data JPA, Microservices, RESTful APIs, Flask",
            "• Frontend Development: React.js, HTML5, CSS3, JavaScript, Tailwind CSS, Bootstrap",
            "• Databases & Deployment: MySQL, Firebase, Vercel, Render",
            "• AI & Machine Learning: Convolutional Neural Networks (CNN/MMCNN), Genetic Algorithms, PCA, Facial Recognition, RAG Architectures",
            "• UI/UX Design & Tools: Figma (Wireframing & Prototyping), Git, GitHub, Postman, IoT Microcontrollers"
        ]
        self.chunks.append(Chunk(
            chunk_id="skills_all",
            title="Technical Skills & Tech Stack Overview",
            category="skill",
            content="\n".join(all_skills_lines),
            keywords=["skills", "tech stack", "technologies", "tools", "programming", "languages", "java", "python", "spring boot", "react", "mysql", "figma"]
        ))

    def _load_experience(self):
        path = os.path.join(self.knowledge_dir, "experience.json")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for exp in data.get("internships", []):
            company = exp.get("company")
            role = exp.get("role")
            tech = ", ".join(exp.get("technologies", []))
            contributions = "\n".join([f"• {c}" for c in exp.get("key_contributions", [])])

            content = f"Company: {company}\n" \
                      f"Role: {role}\n" \
                      f"Location: {exp.get('location')}\n" \
                      f"Duration: {exp.get('duration')}\n" \
                      f"Technologies Used: {tech}\n" \
                      f"Summary: {exp.get('summary')}\n" \
                      f"Key Responsibilities & Deliverables:\n{contributions}"

            cid = f"exp_{company.lower().split()[0]}"
            kw = ["internship", "experience", "work", "job", "career", company.lower(), role.lower()] + [t.lower() for t in exp.get("technologies", [])]
            links = {}
            if "Elysium" in company:
                links["live_demo"] = "https://asset-frontend-xi.vercel.app/"
                links["github"] = "https://github.com/Arunkumar1234KR/Asset_Management_System"

            self.chunks.append(Chunk(
                chunk_id=cid,
                title=f"Experience: {role} at {company}",
                category="experience",
                content=content,
                keywords=kw,
                links=links,
                metadata=exp
            ))

        # Consolidated experience overview
        exp_overview_lines = [
            "Arunkumar's Industry Internships & Practical Experience:",
            "1. Elysium Technologies, Madurai (Full Stack Developer Intern, Feb 2024 – Mar 2024):",
            "   • Engineered the enterprise Asset Management System using Spring Boot, React.js, and MySQL.",
            "   • Built secure REST APIs with Spring Data JPA and deployed client on Vercel and backend on Render.",
            "2. Reccsar, Madurai (Web Development Intern, June 2023 – July 2023):",
            "   • Developed responsive e-commerce web modules using HTML, CSS, JavaScript, and PHP/MySQL integration.",
            "   • Optimized client-side page load speed, cross-device responsiveness, and database query latency."
        ]
        self.chunks.append(Chunk(
            chunk_id="exp_overview",
            title="Work Experience & Industry Internships Overview",
            category="experience",
            content="\n".join(exp_overview_lines),
            keywords=["experience", "internship", "internships", "work", "job", "career", "elysium", "reccsar", "companies"],
            links={"elysium_demo": "https://asset-frontend-xi.vercel.app/", "github": "https://github.com/Arunkumar1234KR"}
        ))

    def _load_projects(self):
        path = os.path.join(self.knowledge_dir, "projects.json")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for proj in data.get("projects", []):
            pid = proj.get("id")
            title = proj.get("title")
            cat = proj.get("category")
            tech = ", ".join(proj.get("technologies", []))
            features = "\n".join([f"• {f}" for f in proj.get("key_features", [])])

            content = f"Project: {title}\n" \
                      f"Category: {cat}\n" \
                      f"Timeline: {proj.get('date')}\n" \
                      f"Overview: {proj.get('overview')}\n" \
                      f"Technologies: {tech}\n"
            if "problem_solved" in proj:
                content += f"Problem Addressed: {proj.get('problem_solved')}\n"
            content += f"Core Capabilities & Features:\n{features}\n"
            if "achievement" in proj:
                content += f"Honors & Recognition: {proj.get('achievement')}\n"
            if "publication" in proj:
                content += f"Publication: {proj.get('publication')}\n"
            if "impact" in proj:
                content += f"Impact & Results: {proj.get('impact')}\n"

            links = {}
            if "live_demo" in proj:
                links["live_demo"] = proj.get("live_demo")
            if "github" in proj:
                links["github"] = proj.get("github")
            if pid == "fruit_health_analysis":
                links["ieee"] = "https://ieeexplore.ieee.org/abstract/document/11012639"
            elif pid == "ml_smart_epds":
                links["ieee"] = "https://ieeexplore.ieee.org/abstract/document/11382690"

            kw = ["project", "build", "developed", title.lower(), pid] + [t.lower() for t in proj.get("technologies", [])]
            if "nebulizer" in title.lower():
                kw += ["apn", "nebulizer", "pulse", "sensor", "iot", "asthma", "isti", "award", "contest", "medical", "hardware"]
            elif "epds" in pid:
                kw += ["epds", "pds", "ration", "ration shop", "fair price", "facial recognition", "crowd detection", "public welfare"]
            elif "fruit" in pid:
                kw += ["fruit", "agriculture", "mmcnn", "cnn", "pca", "genetic algorithm", "disease", "spectral"]
            elif "asset" in pid:
                kw += ["asset management", "lms", "inventory", "spring boot", "react", "mysql", "vercel"]
            elif "jewelry" in pid:
                kw += ["jewelry", "jewellery", "figma", "ui", "ux", "prototype", "design", "wireframe"]

            self.chunks.append(Chunk(
                chunk_id=f"proj_{pid}",
                title=f"Project: {title}",
                category="project",
                content=content,
                keywords=kw,
                links=links,
                metadata=proj
            ))

        # Consolidated projects overview
        proj_overview_lines = [
            "Arunkumar's Top 5 Engineering & Research Projects:",
            "1. Automated Portable Nebulization System (APN) [IoT & Smart Healthcare] - Won 3rd Place in National Project Contest (ISTI).",
            "2. ML-Based Smart e-PDS (Public Distribution System) [AI/ML & Social Good] - IEEE Published Paper (Face recognition, queue management, UPI).",
            "3. ML-Based Fruit Health and Quality Analysis System [Computer Vision & Deep Learning] - IEEE Published Paper (MMCNN + PCA + Genetic Algorithms).",
            "4. Enterprise Asset Management System [Full Stack Web Dev] - Built with Spring Boot, React.js, and MySQL (Live on Vercel).",
            "5. Jewelry E-Commerce UX Design [UI/UX & Product Design] - High-fidelity interactive Figma prototype."
        ]
        self.chunks.append(Chunk(
            chunk_id="proj_overview",
            title="Projects Overview & Portfolio Highlights",
            category="project",
            content="\n".join(proj_overview_lines),
            keywords=["projects", "project", "portfolio", "built", "developed", "showcase", "list projects", "top projects", "all projects"],
            links={
                "portfolio": "https://krarun-portfolio.netlify.app/",
                "asset_management_demo": "https://asset-frontend-xi.vercel.app/",
                "github": "https://github.com/Arunkumar1234KR"
            }
        ))

    def _load_publications(self):
        path = os.path.join(self.knowledge_dir, "publications.json")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for pub in data.get("publications", []):
            title = pub.get("title")
            url = pub.get("url")
            date = pub.get("publication_date")
            publisher = pub.get("publisher")
            abstract = pub.get("abstract")
            keywords = pub.get("keywords", [])

            content = f"Research Paper: {title}\n" \
                      f"Publisher: {publisher}\n" \
                      f"Published: {date}\n" \
                      f"Official IEEE Xplore Link: {url}\n" \
                      f"Abstract / Summary:\n{abstract}\n" \
                      f"Keywords: {', '.join(keywords)}"

            kw = ["paper", "publication", "research", "ieee", "conference", "xplore", "author", "journal", title.lower()] + [k.lower() for k in keywords]
            self.chunks.append(Chunk(
                chunk_id=f"pub_{title[:20].lower().replace(' ', '_')}",
                title=f"Research Publication: {title}",
                category="publication",
                content=content,
                keywords=kw,
                links={"ieee_paper": url},
                metadata=pub
            ))

    def _load_certifications(self):
        path = os.path.join(self.knowledge_dir, "certifications.json")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Certifications chunk
        cert_lines = ["Industry Certifications & Credentials Earned by Arunkumar:"]
        cert_kw = ["certifications", "certificate", "credentials", "course", "licensed", "wipro", "ibm", "tcs", "microsoft", "nptel", "pmi"]
        for c in data.get("certifications", []):
            cert_lines.append(f"• {c.get('title')} (Issued by: {c.get('issuer')}): {c.get('focus')}")
            cert_kw.append(c.get('issuer').lower())

        self.chunks.append(Chunk(
            chunk_id="certs_list",
            title="Certifications & Professional Credentials",
            category="certification",
            content="\n".join(cert_lines),
            keywords=cert_kw,
            metadata={"certifications": data.get("certifications", [])}
        ))

        # Achievements chunk
        ach_lines = ["Key Honors & Competitive Achievements:"]
        ach_kw = ["achievements", "awards", "honors", "prizes", "competitions", "contest", "isti", "tce", "rank", "place"]
        for a in data.get("achievements", []):
            ach_lines.append(f"• {a.get('title')} ({a.get('institution', a.get('sponsor', ''))}): {a.get('description')}")

        self.chunks.append(Chunk(
            chunk_id="achievements_list",
            title="Awards & Competitive Achievements",
            category="certification",
            content="\n".join(ach_lines),
            keywords=ach_kw,
            metadata={"achievements": data.get("achievements", [])}
        ))

    def _load_faq(self):
        path = os.path.join(self.knowledge_dir, "faq.json")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for i, item in enumerate(data.get("faq", [])):
            q = item.get("question")
            ans = item.get("answer")
            kw = ["faq", "question", q.lower()] + [w.lower() for w in q.split() if len(w) > 3]

            self.chunks.append(Chunk(
                chunk_id=f"faq_{i+1}",
                title=f"FAQ: {q}",
                category="faq",
                content=f"Question: {q}\n\nAnswer: {ans}",
                keywords=kw,
                metadata={"question": q, "answer": ans}
            ))

    def _load_live_portfolio(self):
        path = os.path.join(self.knowledge_dir, "live_portfolio.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            last_synced = data.get("last_synced", "Recently")
            pages = data.get("pages", [])
            for page in pages:
                slug = page.get("page_slug", "home")
                title = page.get("title", slug)
                text = page.get("text_preview", "")
                url = page.get("url", "")
                if len(text) < 50:
                    continue

                self.chunks.append(Chunk(
                    chunk_id=f"web_{slug}",
                    title=f"Live Portfolio Sync: {title} ({slug})",
                    category="live_web",
                    content=f"Source: {url} (Live synced {last_synced})\nContent Summary:\n{text[:1800]}",
                    keywords=["live", "portfolio", "latest", "updated", "website", slug],
                    links={"page_url": url},
                    metadata={"source": url, "synced_at": last_synced}
                ))
        except Exception as e:
            print(f"[RAGEngine] Warning: could not parse live_portfolio.json: {e}")

    def retrieve(self, query: str, top_k: int = 4) -> List[Tuple[Chunk, float]]:
        """Retrieve most relevant knowledge chunks using hybrid BM25 + exact entity matching."""
        if not self.chunks:
            return []

        # Normalize query (fixes typos like 'perrentage' -> 'percentage', '10 th' -> '10th')
        norm_query = clean_and_normalize(query)
        norm_lower = norm_query.lower()
        query_lower = query.lower()
        query_tokens = self.tokenize(norm_query)

        # Domain query expansion & synonym matching
        expanded_tokens = list(query_tokens)
        synonyms_map = {
            "gpa": ["cgpa", "marks", "grades", "percentage"],
            "score": ["cgpa", "marks", "percentage"],
            "10th": ["sslc", "secondary", "85.40", "school", "percentage"],
            "tenth": ["10th", "sslc", "secondary", "school", "percentage"],
            "sslc": ["10th", "secondary", "school", "85.40", "percentage"],
            "12th": ["hsc", "higher", "secondary", "92.50", "school", "percentage"],
            "twelfth": ["12th", "hsc", "school", "percentage"],
            "hsc": ["12th", "higher", "secondary", "school", "92.50", "percentage"],
            "school": ["muthuthevar", "mukkulathore", "hsc", "sslc", "10th", "12th"],
            "college": ["velammal", "vcet", "btech", "degree", "honors", "8.84"],
            "work": ["experience", "internship", "job", "elysium", "reccsar"],
            "company": ["elysium", "reccsar"],
            "paper": ["publication", "ieee", "xplore", "research"],
            "research": ["publication", "ieee", "mmcnn", "smart_epds"],
            "nebulizer": ["apn", "pulse", "sensor", "isti", "asthma"],
            "epds": ["ration", "smart_epds", "facial", "recognition"],
            "ration": ["epds", "smart_epds", "pds"],
            "hire": ["why_hire", "strengths", "contact", "email", "phone", "resume"],
            "reach": ["contact", "phone", "email", "linkedin"],
            "call": ["phone", "mobile", "number"],
            "cv": ["resume", "download", "link"],
            "awards": ["achievements", "isti", "tce", "contest", "prize"]
        }
        for word in query_tokens:
            if word in synonyms_map:
                expanded_tokens.extend(synonyms_map[word])

        scores: Dict[int, float] = {}

        # 1. BM25 inverted index scoring
        for token in expanded_tokens:
            if token in self.inverted_index:
                token_idf = self.idf.get(token, 1.0)
                for doc_idx, bm25_tf in self.inverted_index[token]:
                    scores[doc_idx] = scores.get(doc_idx, 0.0) + (bm25_tf * token_idf)

        # 2. Exact keyword and substring boost
        for idx, chunk in enumerate(self.chunks):
            chunk_kw = chunk.keywords
            # Direct match in keywords
            for q_tok in query_tokens:
                if q_tok in chunk_kw:
                    scores[idx] = scores.get(idx, 0.0) + 3.5

            # Direct match in title
            if any(q_tok in chunk.title.lower() for q_tok in query_tokens):
                scores[idx] = scores.get(idx, 0.0) + 2.5

            # 3. Live web scraping penalty: Keep structured knowledge first unless explicitly asking for live sync
            if chunk.category == "live_web":
                if not any(w in norm_lower for w in ["live sync", "synced", "latest website", "raw site"]):
                    scores[idx] = scores.get(idx, 0.0) - 35.0

            # 4. High-Priority Semantic Entity Triggers
            # Dedicated 10th / SSLC boost
            if any(w in norm_lower for w in ["10th", "tenth", "sslc", "secondary"]):
                if chunk.chunk_id == "edu_10th_sslc":
                    scores[idx] = scores.get(idx, 0.0) + 40.0
                elif chunk.chunk_id == "edu_overview":
                    scores[idx] = scores.get(idx, 0.0) + 15.0

            # Dedicated 12th / HSC boost
            if any(w in norm_lower for w in ["12th", "twelfth", "hsc", "plus two", "+2"]):
                if chunk.chunk_id == "edu_12th_hsc":
                    scores[idx] = scores.get(idx, 0.0) + 40.0
                elif chunk.chunk_id == "edu_overview":
                    scores[idx] = scores.get(idx, 0.0) + 15.0

            # Dedicated College / CGPA / Degree boost
            if any(w in norm_lower for w in ["college", "university", "vcet", "cgpa", "btech", "degree", "velammal"]):
                if chunk.chunk_id == "edu_college":
                    scores[idx] = scores.get(idx, 0.0) + 40.0
                elif chunk.chunk_id == "edu_overview":
                    scores[idx] = scores.get(idx, 0.0) + 15.0

            # Complete Educational Record boost
            if any(w in norm_lower for w in ["education", "academics", "academic record", "qualifications", "schooling"]):
                if chunk.chunk_id == "edu_overview":
                    scores[idx] = scores.get(idx, 0.0) + 35.0

            # Strengths & Why Hire boost (handles both 'strength' singular and 'strengths' plural)
            if any(w in norm_lower for w in ["strength", "strengths", "strong", "capabilities", "good at", "strong points"]):
                if chunk.chunk_id in ["profile_strengths", "faq_1"]:
                    scores[idx] = scores.get(idx, 0.0) + 45.0

            # Weaknesses boost
            if any(w in norm_lower for w in ["weakness", "weaknesses", "improvement", "areas to improve"]):
                if "weakness" in chunk.title.lower() or "weakness" in chunk.content.lower():
                    scores[idx] = scores.get(idx, 0.0) + 45.0

            # Projects Listing boost
            if any(w in norm_lower for w in ["project", "projects", "built", "portfolio", "develop", "created"]):
                if chunk.chunk_id == "proj_overview":
                    scores[idx] = scores.get(idx, 0.0) + 40.0

            # Internships & Work Experience boost
            if any(w in norm_lower for w in ["internship", "internships", "experience", "work", "job", "career", "company", "companies"]):
                if chunk.chunk_id in ["exp_overview", "exp_elysium", "exp_reccsar"]:
                    scores[idx] = scores.get(idx, 0.0) + 40.0

            # Location & Residence boost
            if any(w in norm_lower for w in ["location", "live", "where do you live", "where are you", "city", "address", "native", "madurai", "thirunagar"]):
                if chunk.chunk_id in ["profile_bio", "profile_contact"]:
                    scores[idx] = scores.get(idx, 0.0) + 40.0

            # Technical Skills & Tech Stack boost
            if any(w in norm_lower for w in ["skill", "skills", "tech stack", "technologies", "tools", "programming", "languages"]):
                if chunk.chunk_id in ["skills_all", "faq_5"] or chunk.category == "skill":
                    scores[idx] = scores.get(idx, 0.0) + 35.0

            # Contact & Resume boost
            if any(w in norm_lower for w in ["phone", "call", "mobile", "whatsapp", "email", "reach", "contact", "hire me", "hire him", "hire arunkumar"]):
                if chunk.chunk_id == "profile_contact":
                    scores[idx] = scores.get(idx, 0.0) + 40.0
            if any(w in norm_lower for w in ["resume", "cv", "download resume", "pdf"]):
                if chunk.chunk_id == "profile_contact":
                    scores[idx] = scores.get(idx, 0.0) + 40.0

            # Specific Projects boost
            if ("elysium" in norm_lower) and "elysium" in chunk.chunk_id:
                scores[idx] = scores.get(idx, 0.0) + 30.0
            if ("reccsar" in norm_lower) and "reccsar" in chunk.chunk_id:
                scores[idx] = scores.get(idx, 0.0) + 30.0
            if ("nebulizer" in norm_lower or "apn" in norm_lower) and "nebulizer" in chunk.chunk_id:
                scores[idx] = scores.get(idx, 0.0) + 35.0
            if ("epds" in norm_lower or "ration" in norm_lower) and "smart_epds" in chunk.chunk_id:
                scores[idx] = scores.get(idx, 0.0) + 35.0
            if ("fruit" in norm_lower or "mmcnn" in norm_lower) and "fruit" in chunk.chunk_id:
                scores[idx] = scores.get(idx, 0.0) + 35.0
            if ("asset" in norm_lower) and "asset" in chunk.chunk_id:
                scores[idx] = scores.get(idx, 0.0) + 35.0
            if ("jewelry" in norm_lower or "jewellery" in norm_lower) and "jewelry" in chunk.chunk_id:
                scores[idx] = scores.get(idx, 0.0) + 35.0

            # Publications / IEEE boost
            if any(w in norm_lower for w in ["ieee", "paper", "papers", "publication", "publications", "research", "xplore"]):
                if chunk.category == "publication" or "smart_epds" in chunk.chunk_id or "faq_6" in chunk.chunk_id:
                    scores[idx] = scores.get(idx, 0.0) + 35.0

            # Certifications & Credentials boost
            if any(w in norm_lower for w in ["certif", "certificate", "credentials", "course", "licensed"]):
                if chunk.chunk_id == "certs_list":
                    scores[idx] = scores.get(idx, 0.0) + 35.0

            # Awards & Achievements boost
            if any(w in norm_lower for w in ["award", "awards", "achievement", "achievements", "honor", "honors", "contest", "prize", "isti"]):
                if chunk.chunk_id in ["achievements_list", "faq_7"]:
                    scores[idx] = scores.get(idx, 0.0) + 35.0

            # Introduction & Persona boost
            if any(w in norm_lower for w in ["who are you", "tell me about yourself", "introduce", "who is arun", "about you", "bio"]):
                if chunk.chunk_id == "profile_bio":
                    scores[idx] = scores.get(idx, 0.0) + 35.0

        if not scores:
            # Fallback to general bio and skills
            default_chunks = [c for c in self.chunks if c.chunk_id in ["profile_bio", "skill_backend_development", "profile_contact"]]
            return [(c, 1.0) for c in default_chunks[:top_k]]

        # Sort by score descending
        sorted_indices = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_results = [(self.chunks[idx], score) for idx, score in sorted_indices[:top_k]]
        return top_results

    def reindex(self) -> Dict[str, Any]:
        """Trigger dynamic re-indexing of all knowledge documents."""
        self.index_knowledge()
        return {
            "status": "success",
            "total_chunks": len(self.chunks),
            "categories": list(set(c.category for c in self.chunks))
        }

if __name__ == "__main__":
    engine = RAGEngine()
    test_queries = [
        "What is Arun's CGPA?",
        "Tell me about his Nebulizer project",
        "What did he do at Elysium Technologies?",
        "Does Arun have any IEEE papers?",
        "How can I contact or hire Arunkumar?",
        "What are his technical skills in Java and Spring Boot?"
    ]
    for q in test_queries:
        print(f"\nQ: {q}")
        results = engine.retrieve(q, top_k=2)
        for chunk, score in results:
            print(f"  -> [{chunk.category.upper()}] {chunk.title} (Score: {score:.2f})")
