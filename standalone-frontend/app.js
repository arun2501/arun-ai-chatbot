/**
 * standalone-frontend/app.js
 * Standalone Frontend controller for Arunkumar K R's Personal AI Chatbot.
 */

class ArunAIChatStandalone {
    constructor() {
        this.chatMessages = document.getElementById('chatMessages');
        this.welcomeHero = document.getElementById('welcomeHero');
        this.typingIndicator = document.getElementById('typingIndicator');
        this.userInput = document.getElementById('userInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.micBtn = document.getElementById('micBtn');
        this.syncBtn = document.getElementById('syncBtn');
        this.toastNotification = document.getElementById('toastNotification');

        this.backendUrl = localStorage.getItem('arun_backend_url') || (window.location.origin.includes('http') ? window.location.origin : 'http://127.0.0.1:5000');
        this.recognition = null;
        this.isRecording = false;

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupSpeechRecognition();

        if (this.userInput) this.userInput.focus();
    }

    setupEventListeners() {
        this.sendBtn.addEventListener('click', () => this.sendMessage());

        this.userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        this.userInput.addEventListener('input', () => {
            this.userInput.style.height = 'auto';
            this.userInput.style.height = (this.userInput.scrollHeight) + 'px';
        });

        document.querySelectorAll('.prompt-chip').forEach(chip => {
            chip.addEventListener('click', () => {
                const prompt = chip.getAttribute('data-prompt');
                if (prompt) {
                    this.userInput.value = prompt;
                    this.sendMessage();
                }
            });
        });

        if (this.syncBtn) {
            this.syncBtn.addEventListener('click', () => this.syncPortfolio());
        }
    }

    sendMessage() {
        const text = this.userInput.value.trim();
        if (!text) return;

        if (this.welcomeHero) this.welcomeHero.style.display = 'none';

        this.appendUserMessage(text);
        this.userInput.value = '';
        this.userInput.style.height = 'auto';

        this.showTyping(true);
        this.scrollToBottom();

        const url = `${this.backendUrl.replace(/\/$/, '')}/predict`;

        fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text })
        })
        .then(res => {
            if (!res.ok) throw new Error(`HTTP error ${res.status}`);
            return res.json();
        })
        .then(data => {
            this.showTyping(false);
            this.appendBotMessage(data);
            this.scrollToBottom();
        })
        .catch(err => {
            console.error('Chat error:', err);
            this.showTyping(false);
            this.appendBotMessage({
                answer: "Unable to connect to the backend server at `" + this.backendUrl + "`. Please ensure the Python backend (`python app.py`) is running, or set your deployed URL in Settings!",
                engine: "Connection Error",
                sources: [],
                suggested_chips: ["📞 Contact Arunkumar directly", "📄 Download CV"]
            });
            this.scrollToBottom();
        });
    }

    appendUserMessage(text) {
        const row = document.createElement('div');
        row.className = 'message-row user-row';
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.textContent = text;
        row.appendChild(bubble);
        this.chatMessages.appendChild(row);
    }

    appendBotMessage(data) {
        const row = document.createElement('div');
        row.className = 'message-row bot-row';

        const avatar = document.createElement('img');
        avatar.className = 'bot-avatar-mini';
        avatar.src = 'images/arun1.jpg';
        avatar.alt = 'Arun AI';
        avatar.onerror = () => { avatar.src = 'https://ui-avatars.com/api/?name=Arun+Kumar&background=0082ca&color=fff'; };

        const contentCol = document.createElement('div');
        contentCol.style.display = 'flex';
        contentCol.style.flexDirection = 'column';
        contentCol.style.width = '100%';

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';

        let rawHtml = '';
        if (window.marked && typeof window.marked.parse === 'function') {
            rawHtml = window.marked.parse(data.answer || '');
        } else {
            rawHtml = (data.answer || '').replace(/\n/g, '<br>');
        }
        bubble.innerHTML = rawHtml;

        bubble.querySelectorAll('a').forEach(a => {
            a.setAttribute('target', '_blank');
            a.setAttribute('rel', 'noopener noreferrer');
        });

        // Bot meta bar (actions only)
        const metaBar = document.createElement('div');
        metaBar.className = 'bot-meta-bar';

        const actions = document.createElement('div');
        actions.className = 'message-actions';

        const copyBtn = document.createElement('button');
        copyBtn.className = 'msg-action-btn';
        copyBtn.title = 'Copy Answer';
        copyBtn.innerHTML = '<i class="bi bi-clipboard"></i>';
        copyBtn.addEventListener('click', () => {
            navigator.clipboard.writeText(data.answer).then(() => {
                copyBtn.innerHTML = '<i class="bi bi-check2" style="color: #10b981;"></i>';
                setTimeout(() => { copyBtn.innerHTML = '<i class="bi bi-clipboard"></i>'; }, 2000);
            });
        });
        actions.appendChild(copyBtn);

        if ('speechSynthesis' in window) {
            const ttsBtn = document.createElement('button');
            ttsBtn.className = 'msg-action-btn';
            ttsBtn.title = 'Read Aloud (Voice)';
            ttsBtn.innerHTML = '<i class="bi bi-volume-up"></i>';
            ttsBtn.addEventListener('click', () => {
                window.speechSynthesis.cancel();
                const cleanSpeech = (data.answer || '').replace(/[\#\*\_\`\[\]\(\)]/g, ' ');
                const utterance = new SpeechSynthesisUtterance(cleanSpeech);
                utterance.rate = 1.0;
                window.speechSynthesis.speak(utterance);
            });
            actions.appendChild(ttsBtn);
        }

        metaBar.appendChild(actions);
        bubble.appendChild(metaBar);
        contentCol.appendChild(bubble);

        if (data.suggested_chips && data.suggested_chips.length > 0) {
            const chipsRow = document.createElement('div');
            chipsRow.className = 'followup-chips-row';

            data.suggested_chips.forEach(chipText => {
                const chip = document.createElement('button');
                chip.className = 'followup-chip';
                chip.textContent = chipText;
                chip.addEventListener('click', () => {
                    this.userInput.value = chipText;
                    this.sendMessage();
                });
                chipsRow.appendChild(chip);
            });
            contentCol.appendChild(chipsRow);
        }

        row.appendChild(avatar);
        row.appendChild(contentCol);
        this.chatMessages.appendChild(row);
    }

    showTyping(show) {
        if (this.typingIndicator) {
            this.typingIndicator.style.display = show ? 'flex' : 'none';
        }
    }

    scrollToBottom() {
        if (this.chatMessages) {
            setTimeout(() => {
                this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
            }, 50);
        }
    }

    setupSpeechRecognition() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            if (this.micBtn) this.micBtn.style.display = 'none';
            return;
        }

        this.recognition = new SpeechRecognition();
        this.recognition.continuous = false;
        this.recognition.interimResults = false;
        this.recognition.lang = 'en-US';

        this.recognition.onstart = () => {
            this.isRecording = true;
            this.micBtn.classList.add('recording');
        };

        this.recognition.onend = () => {
            this.isRecording = false;
            this.micBtn.classList.remove('recording');
        };

        this.recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            if (transcript) {
                this.userInput.value = transcript;
                this.sendMessage();
            }
        };

        this.micBtn.addEventListener('click', () => {
            if (this.isRecording) {
                this.recognition.stop();
            } else {
                this.recognition.start();
            }
        });
    }

    syncPortfolio() {
        const icon = this.syncBtn.querySelector('i');
        if (icon) icon.classList.add('spin');

        const url = `${this.backendUrl.replace(/\/$/, '')}/api/sync`;
        fetch(url, { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            if (icon) icon.classList.remove('spin');
            this.showToast('✓ Successfully refreshed live info from krarun-portfolio.netlify.app!');
        })
        .catch(err => {
            if (icon) icon.classList.remove('spin');
            this.showToast('⚠ Live sync notice: using current indexed knowledge.');
        });
    }

    showToast(msg) {
        if (!this.toastNotification) return;
        this.toastNotification.textContent = msg;
        this.toastNotification.style.display = 'block';
        setTimeout(() => {
            this.toastNotification.style.display = 'none';
        }, 4000);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new ArunAIChatStandalone();
});
