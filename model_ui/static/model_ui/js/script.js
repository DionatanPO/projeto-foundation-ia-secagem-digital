// ── Gestão de Tema (Manual + Sistema) ──
(function () {
    const themeToggle = document.getElementById('themeToggle');
    const themeLabel = document.getElementById('themeLabel');
    const themeIcon = document.getElementById('themeIcon');
    const hljsTheme = document.getElementById('hljs-theme');

    const darkStyles = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/github-dark.min.css';
    const lightStyles = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/github.min.css';

    function setTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);

        if (theme === 'dark') {
            themeLabel.innerText = 'Modo Escuro';
            themeIcon.innerHTML = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>';
            hljsTheme.href = darkStyles;
        } else {
            themeLabel.innerText = 'Modo Claro';
            themeIcon.innerHTML = '<circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>';
            hljsTheme.href = lightStyles;
        }
    }

    // Inicialização
    const savedTheme = localStorage.getItem('theme');
    const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    setTheme(savedTheme || systemTheme);

    themeToggle.addEventListener('click', () => {
        const current = document.documentElement.getAttribute('data-theme');
        setTheme(current === 'dark' ? 'light' : 'dark');
    });

    // Ouvir mudança do sistema se não houver preferência salva
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
        if (!localStorage.getItem('theme')) {
            setTheme(e.matches ? 'dark' : 'light');
        }
    });
})();

function toggleModal(show) {
    const modal = document.getElementById('settingsModal');
    if (show) {
        modal.classList.add('show');
    } else {
        modal.classList.remove('show');
    }
}

const chatArea = document.getElementById('chatArea');
const promptInput = document.getElementById('promptInput');
const sendBtn = document.getElementById('sendBtn');
const micBtn = document.getElementById('micBtn');
const previewStrip = document.getElementById('previewStrip');
const inputBox = document.getElementById('inputBox');
const welcomeHero = document.getElementById('welcomeHero');
let isWaiting = false;
let currentImageBase64 = null;
let firstMessage = true;
let chatHistory = []; // Memória da conversa
let currentAbortController = null;

const paperPlaneIcon = `
    <svg viewBox="0 0 24 24" fill="currentColor">
        <path d="M2,21L23,12L2,3V10L17,12L2,14V21Z" />
    </svg>
`;

const stopIcon = `
    <svg viewBox="0 0 24 24" fill="currentColor">
        <rect x="6" y="6" width="12" height="12" rx="2" />
    </svg>
`;

function setSendButtonStopState() {
    sendBtn.innerHTML = stopIcon;
    sendBtn.classList.add('stop-style');
    sendBtn.classList.remove('active');
    sendBtn.title = "Parar resposta";
}

function resetSendButton() {
    sendBtn.innerHTML = paperPlaneIcon;
    sendBtn.classList.remove('stop-style');
    sendBtn.title = "Enviar mensagem";
    const hasTextOrImg = promptInput.value.trim().length > 0 || !!currentImageBase64;
    sendBtn.classList.toggle('active', hasTextOrImg);
}

function now() {
    return new Date().toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit' });
}

// Preenche o campo de texto ao clicar nos chips e foca no input
window.fillPrompt = function(text) {
    promptInput.value = text;
    promptInput.dispatchEvent(new Event('input'));
    promptInput.focus();
};

function dismissWelcome() {
    if (!firstMessage) return;
    firstMessage = false;
    welcomeHero.classList.add('dismissing');
    setTimeout(() => {
        welcomeHero.style.display = 'none';
        chatArea.style.display = 'block';
    }, 400);
}

// marked config
marked.setOptions({
    highlight: function (code, lang) {
        const language = hljs.getLanguage(lang) ? lang : 'plaintext';
        return hljs.highlight(code, { language }).value;
    }
});

// Speech
let recognition, isRecording = false;
if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SR();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'pt-BR';
    recognition.onresult = e => {
        for (let i = e.resultIndex; i < e.results.length; i++) {
            if (e.results[i].isFinal) {
                promptInput.value += e.results[i][0].transcript;
                promptInput.dispatchEvent(new Event('input'));
            }
        }
    };
    recognition.onend = () => { isRecording = false; micBtn.classList.remove('recording'); };
    recognition.onerror = () => { isRecording = false; micBtn.classList.remove('recording'); };
}

window.toggleVoice = function() {
    if (!recognition) { alert("Navegador não suporta ditado por voz."); return; }
    isRecording ? recognition.stop() : (recognition.start(), isRecording = true, micBtn.classList.add('recording'));
};

window.handleImageUpload = function(event) {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = e => {
        currentImageBase64 = e.target.result.split(',')[1];
        document.getElementById('imagePreview').src = e.target.result;
        previewStrip.style.display = 'flex';
        inputBox.classList.add('has-image');
        if (!isWaiting) sendBtn.classList.add('active');
    };
    reader.readAsDataURL(file);
};

window.removeImage = function() {
    currentImageBase64 = null;
    document.getElementById('imageInput').value = '';
    previewStrip.style.display = 'none';
    inputBox.classList.remove('has-image');
    if (!isWaiting && !promptInput.value.trim()) sendBtn.classList.remove('active');
};

promptInput.addEventListener('input', function () {
    this.style.height = '40px';
    this.style.height = Math.min(this.scrollHeight, 140) + 'px';
    if (!isWaiting) {
        sendBtn.classList.toggle('active', this.value.trim().length > 0 || !!currentImageBase64);
    }
});

promptInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

function thinkBlockHTML(content, label, isExpanded = false) {
    const plain = content.replace(/<[^>]+>/g, '').trim();
    if (!plain) return '';

    return `<div class="think-block ${isExpanded ? 'expanded' : ''}">
        <div class="think-header" onclick="this.closest('.think-block').classList.toggle('expanded'); const txt = this.querySelector('.think-toggle span'); txt.innerText = txt.innerText === 'Ver raciocínio' ? 'Fechar raciocínio' : 'Ver raciocínio';">
            <div class="think-header-left">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"></path>
                    <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"></path>
                </svg>
                ${label}
            </div>
            <div class="think-toggle">
                <span>${isExpanded ? 'Pensando...' : 'Ver raciocínio'}</span>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="6 9 12 15 18 9"></polyline>
                </svg>
            </div>
        </div>
        <div class="think-body">
            <div class="think-scroll">${plain}</div>
        </div>
    </div>`;
}

function renderBotContent(text, isStreaming = true) {
    const cleanText = text.replace(/\[METRICS\].*?\[\/METRICS\]/g, '');
    const isThinkingNow = isStreaming; // Durante o streaming, assumimos que está pensando se a tag estiver aberta

    // 1. Tags <think>
    const thinkMatch = cleanText.match(/<think>([\s\S]*?)(?:<\/think>|$)/);
    if (thinkMatch) {
        const parts = cleanText.split(/<think>|<\/think>/);
        const isFinal = cleanText.includes('</think>');
        return marked.parse(parts[0] || '') +
            thinkBlockHTML(thinkMatch[1].trim(), 'Processo de Raciocínio', !isFinal) +
            marked.parse(parts[2] || '');
    }

    // 2. Padrão "Thinking Process:"
    if (cleanText.toLowerCase().includes('thinking process:')) {
        const markerMatch = cleanText.match(/thinking process:/i);
        const marker = markerMatch[0];
        const markerIndex = cleanText.indexOf(marker);
        const preText = cleanText.substring(0, markerIndex);
        const postText = cleanText.substring(markerIndex + marker.length);

        // Tenta encontrar o fim do pensamento por uma quebra dupla de linha seguida de texto normal
        // (Isso é uma heurística, pois o modelo não envia tag de fechamento)
        let thinkingPart = postText;
        let finalAnswer = "";

        // Se não estiver mais no streaming, tenta separar a resposta final
        if (!isStreaming) {
            const splitPoint = postText.lastIndexOf('\n\n');
            if (splitPoint !== -1) {
                thinkingPart = postText.substring(0, splitPoint);
                finalAnswer = postText.substring(splitPoint);
            }
        }

        return marked.parse(preText) +
            thinkBlockHTML(thinkingPart.trim(), 'Processo de Raciocínio', isStreaming) +
            marked.parse(finalAnswer);
    }

    return marked.parse(cleanText);
}

function extractMetrics(text) {
    const match = text.match(/\[METRICS\](.*?)\[\/METRICS\]/);
    if (match) {
        const [tps, tokens, time] = match[1].split('|');
        return `<div class="perf-badge">
            <span><b>${tps}</b> t/s</span>
            <div class="perf-sep"></div>
            <span><b>${tokens}</b> tokens</span>
            <div class="perf-sep"></div>
            <span><b>${time}s</b></span>
        </div>`;
    }
    return '';
}

function createMsg(text, isUser, imgBase64) {
    const wrap = document.createElement('div');
    wrap.className = `message-wrap ${isUser ? 'user' : 'bot'}`;

    const time = `<span class="msg-time">${now()}</span>`;

    if (isUser) {
        let imgTag = imgBase64 ? `<img src="data:image/jpeg;base64,${imgBase64}" style="max-width:200px;border-radius:8px;margin-bottom:8px;display:block;">` : '';
        wrap.innerHTML = `
        <div class="avatar user">EU</div>
        <div class="msg-body">
            <div class="msg-meta"><span class="msg-name">Você</span>${time}</div>
            <div class="msg-bubble">${imgTag}${text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\n/g, '<br>')}</div>
        </div>`;
    } else {
        wrap.innerHTML = `
        <div class="avatar bot">✦</div>
        <div class="msg-body">
            <div class="msg-meta">
                <span class="msg-name">Secagem AI</span>
                ${time}
                <button class="copy-btn" onclick="copyMsg(this)" title="Copiar resposta">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                    <span>Copiar</span>
                </button>
            </div>
            <div class="msg-bubble"></div>
        </div>`;
        wrap.dataset.rawText = '';
    }
    return wrap;
}

function updateBot(el, chunk) {
    el.dataset.rawText += chunk;
    const bubble = el.querySelector('.msg-bubble');
    bubble.innerHTML = renderBotContent(el.dataset.rawText, true); // isStreaming = true

    // Handle Metrics display outside the bubble
    const metricsHTML = extractMetrics(el.dataset.rawText);
    if (metricsHTML) {
        let badge = el.querySelector('.perf-badge');
        if (!badge) {
            const badgeContainer = document.createElement('div');
            badgeContainer.innerHTML = metricsHTML;
            el.querySelector('.msg-body').appendChild(badgeContainer.firstChild);
        }
    }

    scrollBottom();
}

function finalizeBotMessage(el) {
    const rawText = el.dataset.rawText;

    // Renderiza uma última vez com isStreaming = false para fechar o card e separar a resposta
    el.querySelector('.msg-bubble').innerHTML = renderBotContent(rawText, false);

    chatHistory.push({ role: 'assistant', content: rawText });

    // Mantém apenas as últimas 10 mensagens para não estourar o contexto
    if (chatHistory.length > 10) {
        chatHistory = chatHistory.slice(-10);
    }
}

function showLoader() {
    const wrap = document.createElement('div');
    wrap.className = 'message-wrap bot';
    wrap.id = 'loaderMsg';
    wrap.innerHTML = `
    <div class="avatar bot">✦</div>
    <div class="msg-body">
        <div class="msg-meta"><span class="msg-name">Secagem AI</span></div>
        <div class="msg-bubble">
            <div class="loader-wrap"><span></span><span></span><span></span></div>
        </div>
    </div>`;
    chatArea.appendChild(wrap);
    scrollBottom();
}

function removeLoader() {
    const l = document.getElementById('loaderMsg');
    if (l) l.remove();
}

function scrollBottom() {
    chatArea.scrollTo({ top: chatArea.scrollHeight, behavior: 'smooth' });
}

window.copyMsg = async function(btn) {
    const wrap = btn.closest('.message-wrap');
    const rawText = wrap.dataset.rawText || "";
    // Remove as tags de métricas e o bloco <think> para copiar apenas a resposta limpa
    const cleanText = rawText.replace(/\[METRICS\].*?\[\/METRICS\]/g, '')
        .replace(/<think>[\s\S]*?<\/think>/g, '')
        .replace(/Thinking Process:[\s\S]*?\n\n/gi, '')
        .trim();

    try {
        await navigator.clipboard.writeText(cleanText);
        const span = btn.querySelector('span');
        const oldText = span.innerText;
        span.innerText = 'Copiado!';
        btn.style.borderColor = 'var(--primary)';
        btn.style.color = 'var(--primary)';

        setTimeout(() => {
            span.innerText = oldText;
            btn.style.borderColor = '';
            btn.style.color = '';
        }, 2000);
    } catch (err) {
        console.error('Erro ao copiar:', err);
    }
};

window.sendMessage = async function() {
    if (isWaiting) {
        if (currentAbortController) {
            currentAbortController.abort();
        }
        return;
    }

    const text = promptInput.value.trim();
    if (!text && !currentImageBase64) return;
    dismissWelcome();

    const temperature = parseFloat(document.getElementById('temperature')?.value) || 0.1;

    const sysPromptEl = document.getElementById('systemPrompt');
    const systemPrompt = sysPromptEl ? sysPromptEl.value : null;

    const ragToggleEl = document.getElementById('ragToggle');
    const useRag = ragToggleEl ? ragToggleEl.checked : true;

    const payloadImage = currentImageBase64;

    // Adiciona ao histórico local
    chatHistory.push({ role: 'user', content: text || "[Imagem enviada]" });

    chatArea.appendChild(createMsg(text, true, payloadImage));
    promptInput.value = '';
    promptInput.style.height = '40px';
    removeImage();
    
    isWaiting = true;
    currentAbortController = new AbortController();
    setSendButtonStopState();
    scrollBottom();
    showLoader();

    let botEl = null;

    try {
        const res = await fetch('/api/chat-stream/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: currentAbortController.signal,
            body: JSON.stringify({
                prompt: text || (payloadImage ? "Analise esta imagem." : ""),
                temperature: temperature,
                image_base64: payloadImage,
                history: chatHistory.slice(0, -1),
                system_prompt: systemPrompt,
                use_rag: useRag
            })
        });

        removeLoader();

        if (!res.ok) {
            const err = await res.json();
            const errMsg = createMsg('**Erro do servidor:**\n```json\n' + JSON.stringify(err, null, 2) + '\n```', false);
            chatArea.appendChild(errMsg);
            return;
        }

        botEl = createMsg('', false);
        chatArea.appendChild(botEl);

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            updateBot(botEl, decoder.decode(value, { stream: true }));
        }
        finalizeBotMessage(botEl);
    } catch (err) {
        if (err.name === 'AbortError') {
            console.log('Stream aborted');
            if (botEl) {
                finalizeBotMessage(botEl);
            } else {
                removeLoader();
            }
        } else {
            console.error(err);
            removeLoader();
            const errEl = createMsg('**Erro de conexão:** Não foi possível acessar o streaming.', false);
            chatArea.appendChild(errEl);
        }
    } finally {
        isWaiting = false;
        currentAbortController = null;
        resetSendButton();
        scrollBottom();
    }
};

// --- Model Management ---
async function loadModels() {
    try {
        const response = await fetch('/api/models/');
        const data = await response.json();
        const modelList = document.getElementById('modelList');
        if (!modelList) return;
        modelList.innerHTML = '';

        if (!data.models || data.models.length === 0) {
            modelList.innerHTML = '<div style="font-size: 12px; color: var(--text-dim); padding: 10px;">Nenhum modelo .gguf encontrado em /models</div>';
            return;
        }

        data.models.forEach(modelName => {
            const isActive = modelName === data.current_model;
            const item = document.createElement('div');
            item.className = `model-item ${isActive ? 'active' : ''}`;
            item.onclick = () => {
                if (!item.classList.contains('active') && !item.classList.contains('switching')) {
                    changeModel(modelName, item);
                }
            };

            item.innerHTML = `
                <div class="model-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16">
                        <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8z"></path>
                        <path d="M12 6v6l4 2"></path>
                    </svg>
                </div>
                <div class="model-info">
                    <span class="model-name" title="${modelName}">${modelName}</span>
                    <span class="model-status">${isActive ? 'Em execução' : 'Disponível'}</span>
                </div>
                ${isActive ? `
                <div style="margin-left: auto;">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" width="14" height="14">
                        <polyline points="20 6 9 17 4 12"></polyline>
                    </svg>
                </div>
                ` : ''}
            `;
            modelList.appendChild(item);
        });
    } catch (err) {
        console.error("Erro ao carregar modelos:", err);
    }
}

async function changeModel(modelName, element) {
    if (isWaiting) return;

    // UI Feedback
    const allItems = document.querySelectorAll('.model-item');
    allItems.forEach(i => i.classList.remove('active', 'switching'));
    element.classList.add('switching');

    try {
        const useGpu = document.getElementById('hardwareSelect').value === 'gpu';
        const response = await fetch('/api/switch-model/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_name: modelName, use_gpu: useGpu })
        });

        if (response.ok) {
            await loadModels(); // Atualiza a lista
            updateRAM(); // Atualiza uso de memória
        } else {
            alert("Erro ao trocar o modelo.");
            loadModels();
        }
    } catch (err) {
        console.error("Erro:", err);
        loadModels();
    }
}

window.clearRagStorage = async function() {
    const btn = document.getElementById('clearRagBtn');
    if (!btn || btn.disabled) return;

    const btnText = btn.querySelector('span');
    const originalText = btnText.innerText;

    btn.disabled = true;
    btn.style.opacity = '0.7';
    btnText.innerText = 'Limpando & Recarregando...';

    try {
        const response = await fetch('/api/clear-rag/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        if (response.ok) {
            btnText.innerText = 'Storage Recriado com Sucesso!';
            btn.style.borderColor = 'var(--primary)';
            btn.style.color = 'var(--primary)';
            setTimeout(() => {
                btnText.innerText = originalText;
                btn.disabled = false;
                btn.style.opacity = '';
                btn.style.borderColor = '';
                btn.style.color = '';
            }, 3000);
        } else {
            alert("Falha ao limpar o Banco Vetorial.");
            btnText.innerText = originalText;
            btn.disabled = false;
            btn.style.opacity = '';
        }
    } catch (err) {
        console.error("Erro ao limpar RAG:", err);
        alert("Erro ao conectar ao servidor para limpar RAG.");
        btnText.innerText = originalText;
        btn.disabled = false;
        btn.style.opacity = '';
    }
};

window.unloadCurrentModel = async function() {
    const btn = document.getElementById('unloadModelBtn');
    if (!btn || btn.disabled) return;

    const btnText = btn.querySelector('span');
    const originalText = btnText.innerText;

    btn.disabled = true;
    btn.style.opacity = '0.7';
    btnText.innerText = 'Descarregando...';

    try {
        const response = await fetch('/api/unload-model/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        if (response.ok) {
            btnText.innerText = 'Modelo Descarregado!';
            btn.style.borderColor = 'rgba(219, 68, 85, 0.4)';
            btn.style.background = 'rgba(219, 68, 85, 0.15)';

            await loadModels(); // Atualiza a lista de modelos
            updateRAM(); // Atualiza uso de memória

            setTimeout(() => {
                btnText.innerText = originalText;
                btn.disabled = false;
                btn.style.opacity = '';
                btn.style.borderColor = '';
                btn.style.background = '';
            }, 3000);
        } else {
            alert("Falha ao descarregar o modelo.");
            btnText.innerText = originalText;
            btn.disabled = false;
            btn.style.opacity = '';
        }
    } catch (err) {
        console.error("Erro ao descarregar o modelo:", err);
        alert("Erro ao conectar ao servidor para descarregar o modelo.");
        btnText.innerText = originalText;
        btn.disabled = false;
        btn.style.opacity = '';
    }
};

window.toggleModal = toggleModal;

// Inicializa a interface
loadModels();
updateRAM();
setInterval(updateRAM, 5000);
