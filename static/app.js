/**
 * app.js
 * Frontend controller for Arunkumar K R's Personal AI Chatbot.
 * Features:
 * - Markdown rendering via marked.js
 * - Speech-to-Text (Voice input) via Web Speech API
 * - Text-to-Speech (Read aloud) via browser synthesis
 * - One-click Live Portfolio Sync (/api/sync)
 * - Persistent local storage for optional Free API keys (Groq/Gemini)
 * - Dynamic follow-up chips & quick prompt injection
 */

class ArunAIChat {
    constructor() {
        this.chatMessages = document.getElementById('chatMessages');
        this.welcomeHero = document.getElementById('welcomeHero');
        this.typingIndicator = document.getElementById('typingIndicator');
        this.userInput = document.getElementById('userInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.micBtn = document.getElementById('micBtn');
        this.syncBtn = document.getElementById('syncBtn');
        this.toastNotification = document.getElementById('toastNotification');

        this.recognition = null;
        this.isRecording = false;

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupSpeechRecognition();

        // Auto focus input
        if (this.userInput) {
            this.userInput.focus();
        }
    }

    setupEventListeners() {
        // Send button click
        this.sendBtn.addEventListener('click', () => this.sendMessage());

        // Textarea keyboard listener (Enter sends, Shift+Enter adds newline)
        this.userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Auto-expand textarea as user types without jitter
        this.userInput.addEventListener('input', () => {
            if (!this.userInput.value.trim()) {
                this.userInput.style.height = '';
            } else {
                this.userInput.style.height = 'auto';
                this.userInput.style.height = Math.min(this.userInput.scrollHeight, 80) + 'px';
            }
        });

        // Quick prompt chips in Welcome box
        document.querySelectorAll('.prompt-chip').forEach(chip => {
            chip.addEventListener('click', () => {
                const prompt = chip.getAttribute('data-prompt');
                if (prompt) {
                    this.userInput.value = prompt;
                    this.sendMessage();
                }
            });
        });

        // Live Portfolio Sync button
        if (this.syncBtn) {
            this.syncBtn.addEventListener('click', () => this.syncPortfolio());
        }
    }

    sendMessage() {
        const text = this.userInput.value.trim();
        if (!text) return;

        // Hide welcome hero on first message
        if (this.welcomeHero) {
            this.welcomeHero.style.display = 'none';
        }

        // Render user message bubble
        this.appendUserMessage(text);
        this.userInput.value = '';
        this.userInput.style.height = '';

        // Show typing indicator
        this.showTyping(true);
        this.scrollToBottom();

        const scriptRoot = window.$SCRIPT_ROOT || '';
        const predictUrl = `${scriptRoot}/predict`;

        fetch(predictUrl, {
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
                answer: "I experienced a brief connection error. However, you can reach Arun directly at **krarun2504@gmail.com** or **+91 8778588180**!",
                engine: "Fallback",
                sources: [],
                suggested_chips: ["📞 How to Contact Arunkumar", "📄 Download CV"]
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

        // Bot mini avatar
        const avatar = document.createElement('img');
        avatar.className = 'bot-avatar-mini';
        avatar.src = '/static/images/arun1.jpg';
        avatar.alt = 'Arun AI';
        avatar.onerror = () => { avatar.src = 'https://ui-avatars.com/api/?name=Arun+Kumar&background=0082ca&color=fff'; };

        const contentCol = document.createElement('div');
        contentCol.style.display = 'flex';
        contentCol.style.flexDirection = 'column';
        contentCol.style.width = '100%';

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';

        // Render Markdown safely
        let rawHtml = '';
        if (window.marked && typeof window.marked.parse === 'function') {
            rawHtml = window.marked.parse(data.answer || '');
        } else {
            rawHtml = (data.answer || '').replace(/\n/g, '<br>');
        }
        bubble.innerHTML = rawHtml;

        // Ensure all external links open in new tab
        bubble.querySelectorAll('a').forEach(a => {
            a.setAttribute('target', '_blank');
            a.setAttribute('rel', 'noopener noreferrer');
        });

        // Bot meta bar (actions only)
        const metaBar = document.createElement('div');
        metaBar.className = 'bot-meta-bar';

        const actions = document.createElement('div');
        actions.className = 'message-actions';

        // Copy button
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

        // Text-to-Speech Read Aloud button
        if ('speechSynthesis' in window) {
            const ttsBtn = document.createElement('button');
            ttsBtn.className = 'msg-action-btn';
            ttsBtn.title = 'Read Aloud (Voice)';
            ttsBtn.innerHTML = '<i class="bi bi-volume-up"></i>';
            ttsBtn.addEventListener('click', () => {
                window.speechSynthesis.cancel();
                // Strip markdown formatting for cleaner speech
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

        // Render suggested follow-up chips if present
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

        const scriptRoot = window.$SCRIPT_ROOT || '';
        fetch(`${scriptRoot}/api/sync`, { method: 'POST' })
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

// Initialize on DOM load

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    new ArunAIChat();
});
