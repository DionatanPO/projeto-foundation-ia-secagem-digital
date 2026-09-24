// ── Gestão de Tema (Manual + Sistema) ──
(function () {
    try {
        const themeToggle = document.getElementById('themeToggle');
        const themeLabel = document.getElementById('themeLabel');
        const themeIcon = document.getElementById('themeIcon');
        const hljsTheme = document.getElementById('hljs-theme');
        if (!themeToggle || !themeLabel || !themeIcon) return;

        const darkStyles = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/github-dark.min.css';
        const lightStyles = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/github.min.css';

        function setTheme(theme) {
            try {
                document.documentElement.setAttribute('data-theme', theme);
                localStorage.setItem('theme', theme);
                if (theme === 'dark') {
                    themeLabel.innerText = 'Modo Escuro';
                    themeIcon.innerHTML = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>';
                    if (hljsTheme) hljsTheme.href = darkStyles;
                } else {
                    themeLabel.innerText = 'Modo Claro';
                    themeIcon.innerHTML = '<circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>';
                    if (hljsTheme) hljsTheme.href = lightStyles;
                }
            } catch (e) { console.warn('theme:', e); }
        }

        // Inicialização
        let savedTheme = null;
        try { savedTheme = localStorage.getItem('theme'); } catch (e) {}
        let systemTheme = 'dark';
        try { systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'; } catch (e) {}
        setTheme(savedTheme || systemTheme);

        themeToggle.addEventListener('click', () => {
            const current = document.documentElement.getAttribute('data-theme');
            setTheme(current === 'dark' ? 'light' : 'dark');
        });

        // Ouvir mudança do sistema se não houver preferência salva
        try {
            window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
                try { if (!localStorage.getItem('theme')) setTheme(e.matches ? 'dark' : 'light'); } catch (err) {}
            });
        } catch (e) {}
    } catch (e) { console.warn('theme init falhou:', e); }
})();

function toggleModal(show) {
    const modal = document.getElementById('settingsModal');
    if (show) {
        modal.classList.add('show');
        try { updateHistoryCountInfo(); } catch (e) {}
    } else {
        modal.classList.remove('show');
    }
}

window.switchTab = function(tabId, btn) {
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.settings-tab').forEach(t => t.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    btn.classList.add('active');
};

const chatArea = document.getElementById('chatArea');
const promptInput = document.getElementById('promptInput');
const sendBtn = document.getElementById('sendBtn');
const micBtn = document.getElementById('micBtn');
const previewStrip = document.getElementById('previewStrip');
const inputBox = document.getElementById('inputBox');
const welcomeHero = document.getElementById('welcomeHero');
let isWaiting = false;
let currentImageBase64 = null;
let currentAbortController = null;

// ── Histórico de conversas (persistido na API, sem localStorage) ──
const CONV_API = '/api/conversations';
let conversations = []; // cache da lista: [{id, title, created_at, updated_at, message_count, messages?}]
let currentConversationId = null;
let chatHistory = []; // alias backend: derivado da conversa atual [{role, content}]
let firstMessage = true;

function getCurrentConversation() {
    return conversations.find(c => c.id === currentConversationId) || null;
}

function getConvMessages(conv) {
    return (conv && Array.isArray(conv.messages)) ? conv.messages : [];
}

function getConvUpdatedAt(conv) {
    return conv.updated_at || conv.updatedAt || conv.created_at || conv.createdAt || Date.now();
}

function getConvMsgCount(conv) {
    if (Array.isArray(conv.messages)) return conv.messages.length;
    return conv.message_count || 0;
}

async function apiJSON(url, options) {
    const res = await fetch(url, {
        headers: { 'Content-Type': 'application/json' },
        ...(options || {}),
    });
    if (!res.ok) {
        let detail = '';
        try { detail = await res.text(); } catch (e) {}
        throw new Error(`API ${res.status}: ${detail || res.statusText}`);
    }
    if (res.status === 204) return null;
    const text = await res.text();
    return text ? JSON.parse(text) : null;
}

function formatMsgTime(iso) {
    try {
        if (!iso) return now();
        const d = new Date(iso);
        if (isNaN(d.getTime())) return now();
        return d.toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit' });
    } catch (e) { return now(); }
}

function normalizeMessage(m) {
    return {
        role: m.role,
        content: m.content || '',
        image: m.image_base64 || null,
        time: formatMsgTime(m.created_at),
        raw: m.content || '',
    };
}

function normalizeDetail(detail) {
    return {
        id: detail.id,
        title: detail.title || 'Nova conversa',
        created_at: detail.created_at,
        updated_at: detail.updated_at,
        createdAt: detail.created_at,
        updatedAt: detail.updated_at,
        message_count: (detail.messages || []).length,
        messages: (detail.messages || []).map(normalizeMessage),
    };
}

function syncChatHistory() {
    const conv = getCurrentConversation();
    const msgs = getConvMessages(conv);
    firstMessage = !conv || msgs.length === 0;
    // chatHistory = últimas 10 mensagens no formato do backend
    chatHistory = msgs.slice(-10).map(m => ({ role: m.role, content: m.content }));
}

function buildTitle(text) {
    const t = (text || '').replace(/\s+/g, ' ').trim();
    if (!t) return 'Conversa com imagem';
    return t.length > 38 ? t.slice(0, 38) + '…' : t;
}

// Rascunho: currentConversationId === null significa tela inicial
// ("Como posso ajudar hoje?") sem nada criado no servidor ainda.
function visibleConversations() {
    return conversations.filter(c => getConvMsgCount(c) > 0);
}

async function loadStore() {
    const list = document.getElementById('historyList');
    if (list) list.innerHTML = '<div class="history-empty">Carregando conversas…</div>';
    try {
        const data = await apiJSON(CONV_API + '/');
        let all = Array.isArray(data) ? data : [];
        // garante shape mínimo para itens da lista (sem messages ainda)
        all = all.map(c => ({
            ...c,
            createdAt: c.created_at || c.createdAt,
            updatedAt: c.updated_at || c.updatedAt,
        }));
        // mostra no menu SÓ quem já tem mensagem; limpa "Nova conversa" vazias antigas
        const empties = all.filter(c => getConvMsgCount(c) === 0);
        conversations = all.filter(c => getConvMsgCount(c) > 0);
        if (empties.length) {
            empties.forEach(c => {
                apiJSON(`${CONV_API}/${c.id}/`, { method: 'DELETE' }).catch(() => {});
            });
        }
        currentConversationId = conversations.length ? conversations[0].id : null;
        if (currentConversationId != null) {
            await loadConversationDetail(currentConversationId, true);
        }
    } catch (e) {
        console.error('Falha ao carregar histórico da API:', e);
        if (list) list.innerHTML = '<div class="history-empty">Falha ao carregar histórico do servidor</div>';
        conversations = [];
        currentConversationId = null;
    }
    syncChatHistory();
    renderHistoryList();
    renderCurrentMessages();
}

// Cria a conversa no servidor SOMENTE na hora do primeiro envio.
async function ensureConversationForSend(text) {
    let conv = getCurrentConversation();
    if (conv) return conv;
    const created = await apiJSON(CONV_API + '/', {
        method: 'POST',
        body: JSON.stringify({ title: buildTitle(text) }),
    });
    const norm = normalizeDetail(created);
    norm.title = buildTitle(text);
    conversations.unshift(norm);
    currentConversationId = norm.id;
    syncChatHistory();
    renderHistoryList();
    return norm;
}

async function loadConversationDetail(id, silent) {
    const detail = await apiJSON(`${CONV_API}/${id}/`);
    const norm = normalizeDetail(detail);
    const idx = conversations.findIndex(c => c.id === id);
    if (idx >= 0) conversations[idx] = { ...conversations[idx], ...norm };
    else conversations.unshift(norm);
    if (!silent) {
        syncChatHistory();
        renderHistoryList();
        renderCurrentMessages();
    }
    return norm;
}

async function saveUserMessage(text, imgBase64) {
    const conv = getCurrentConversation();
    if (!conv) return;
    const content = text || '[Imagem enviada]';
    // otimista: atualiza UI na hora
    if (!Array.isArray(conv.messages)) conv.messages = [];
    if (conv.messages.length === 0 || conv.title === 'Nova conversa') {
        conv.title = buildTitle(text);
    }
    conv.messages.push({ role: 'user', content, image: imgBase64 || null, time: now(), raw: content });
    conv.updated_at = new Date().toISOString();
    conv.updatedAt = conv.updated_at;
    syncChatHistory();
    renderHistoryList();
    // persiste no servidor
    try {
        await apiJSON(`${CONV_API}/${conv.id}/messages/`, {
            method: 'POST',
            body: JSON.stringify({ role: 'user', content, image_base64: imgBase64 || '' }),
        });
        // re-sincroniza título/contador vindos do servidor sem trocar de tela
        try {
            const detail = await apiJSON(`${CONV_API}/${conv.id}/`);
            const idx = conversations.findIndex(c => c.id === conv.id);
            if (idx >= 0) {
                const localMsgs = getConvMessages(conversations[idx]);
                const norm = normalizeDetail(detail);
                // preserva mensagens otimistas (evita sumiço durante streaming)
                if (norm.messages.length < localMsgs.length) norm.messages = localMsgs;
                norm.message_count = norm.messages.length;
                conversations[idx] = { ...conversations[idx], ...norm };
                syncChatHistory();
                renderHistoryList();
            }
        } catch (e) { console.warn('sync pós-save user falhou:', e); }
    } catch (e) {
        console.error('Falha ao salvar mensagem do usuário na API:', e);
    }
}

async function saveAssistantMessage(rawText) {
    const text = (rawText || '').trim();
    if (!text) return; // nunca salva resposta vazia (erro de stream, abort precoce, etc.)
    const conv = getCurrentConversation();
    if (!conv) return;
    if (!Array.isArray(conv.messages)) conv.messages = [];
    conv.messages.push({ role: 'assistant', content: text, image: null, time: now(), raw: text });
    conv.updated_at = new Date().toISOString();
    conv.updatedAt = conv.updated_at;
    syncChatHistory();
    renderHistoryList();
    // persiste no servidor — o chamador dá await, então fechar o navegador
    // logo após a resposta não perde o salvamento em condições normais
    await apiJSON(`${CONV_API}/${conv.id}/messages/`, {
        method: 'POST',
        body: JSON.stringify({ role: 'assistant', content: text, image_base64: '' }),
    });
}

function renderHistoryList() {
    const list = document.getElementById('historyList');
    if (!list) return;
    const sorted = visibleConversations().sort((a, b) => new Date(getConvUpdatedAt(b)) - new Date(getConvUpdatedAt(a)));
    if (!sorted.length) {
        list.innerHTML = '<div class="history-empty">Nenhuma conversa ainda</div>';
        try { updateHistoryCountInfo(); } catch (e) {}
        return;
    }
    list.innerHTML = '';
    sorted.forEach(conv => {
        const item = document.createElement('div');
        item.className = 'history-item' + (conv.id === currentConversationId ? ' active' : '');
        item.title = conv.title;
        let dateTime = '';
        try {
            dateTime = new Date(getConvUpdatedAt(conv)).toLocaleString('pt-BR', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' });
        } catch (e) { dateTime = ''; }
        item.innerHTML = `
            <div class="history-icon">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                </svg>
            </div>
            <div class="history-text">
                <span class="history-title"></span>
                <span class="history-date">${dateTime}</span>
            </div>
            <button class="history-rename" title="Renomear conversa">✎</button>
            <button class="history-delete" title="Excluir conversa">✕</button>
        `;
        item.querySelector('.history-title').textContent = conv.title;
        item.onclick = () => window.selectConversation(conv.id);
        item.querySelector('.history-rename').onclick = (e) => { e.stopPropagation(); window.startRenameConversation(conv.id, item); };
        item.querySelector('.history-delete').onclick = (e) => { e.stopPropagation(); window.deleteConversation(conv.id); };
        list.appendChild(item);
    });
    try { updateHistoryCountInfo(); } catch (e) {}
}

function renderCurrentMessages() {
    const conv = getCurrentConversation();
    const msgs = getConvMessages(conv);
    chatArea.innerHTML = '';
    if (!conv || msgs.length === 0) {
        // mostra welcome
        welcomeHero.style.display = 'flex';
        welcomeHero.classList.remove('dismissing');
        chatArea.style.display = 'none';
        firstMessage = true;
        return;
    }
    firstMessage = false;
    welcomeHero.style.display = 'none';
    chatArea.style.display = 'block';
    msgs.forEach(m => {
        if (m.role === 'user') {
            chatArea.appendChild(createMsg(m.content, true, m.image));
        } else {
            const el = createMsg('', false);
            el.dataset.rawText = m.raw || m.content;
            el.querySelector('.msg-bubble').innerHTML = renderBotContent(el.dataset.rawText, false);
            const metricsHTML = extractMetrics(el.dataset.rawText);
            if (metricsHTML) {
                const tmp = document.createElement('div');
                tmp.innerHTML = metricsHTML;
                el.querySelector('.msg-body').appendChild(tmp.firstChild);
            }
            chatArea.appendChild(el);
        }
    });
    scrollBottom();
}

window.newConversation = function() {
    if (isWaiting && currentAbortController) currentAbortController.abort();
    // volta p/ a view inicial SEM criar nada no servidor e SEM listar no menu;
    // a conversa só aparece no histórico após o primeiro envio
    currentConversationId = null;
    syncChatHistory();
    renderHistoryList();
    renderCurrentMessages();
    promptInput.value = '';
    promptInput.focus();
    closeSidebarOnMobile();
};

window.selectConversation = async function(id) {
    if (isWaiting) return; // evita trocar no meio do streaming
    if (id === currentConversationId) {
        // garante detalhe carregado mesmo se lista ainda sem messages
        const conv = getCurrentConversation();
        if (conv && !Array.isArray(conv.messages)) {
            try { await loadConversationDetail(id); } catch (e) {}
        }
        return;
    }
    currentConversationId = id;
    renderHistoryList();
    try {
        await loadConversationDetail(id, true);
    } catch (e) {
        console.error('Falha ao abrir conversa:', e);
    }
    syncChatHistory();
    renderHistoryList();
    renderCurrentMessages();
    closeSidebarOnMobile();
};

window.deleteConversation = async function(id) {
    try {
        await apiJSON(`${CONV_API}/${id}/`, { method: 'DELETE' });
    } catch (e) {
        console.error('Falha ao excluir conversa na API:', e);
        return;
    }
    conversations = conversations.filter(c => c.id !== id);
    if (currentConversationId === id) {
        try {
            // recarrega lista do servidor para pegar ordenação/contadores certos
            const data = await apiJSON(CONV_API + '/');
            conversations = (Array.isArray(data) ? data : []).filter(c => getConvMsgCount(c) > 0);
            currentConversationId = conversations.length ? conversations[0].id : null;
            if (currentConversationId != null) {
                await loadConversationDetail(currentConversationId, true);
            }
        } catch (e) { console.error('Falha ao reorganizar após excluir:', e); }
        syncChatHistory();
    }
    renderHistoryList();
    renderCurrentMessages();
};

window.startRenameConversation = function(id, itemEl) {
    const conv = conversations.find(c => c.id === id);
    if (!conv || !itemEl) return;
    const titleSpan = itemEl.querySelector('.history-title');
    if (!titleSpan || itemEl.querySelector('.history-rename-input')) return;
    const input = document.createElement('input');
    input.className = 'history-rename-input';
    input.type = 'text';
    input.value = conv.title || '';
    input.maxLength = 80;
    titleSpan.replaceWith(input);
    input.focus();
    input.select();
    let done = false;
    const finish = async (save) => {
        if (done) return;
        done = true;
        const newTitle = input.value.trim();
        if (save && newTitle && newTitle !== conv.title) {
            await window.renameConversation(id, newTitle);
        } else {
            renderHistoryList();
        }
    };
    input.addEventListener('keydown', (e) => {
        e.stopPropagation();
        if (e.key === 'Enter') finish(true);
        else if (e.key === 'Escape') finish(false);
    });
    input.addEventListener('blur', () => finish(true));
    input.addEventListener('click', (e) => e.stopPropagation());
};

window.renameConversation = async function(id, newTitle) {
    const title = (newTitle || '').trim().slice(0, 80);
    if (!title) { renderHistoryList(); return; }
    try {
        const updated = await apiJSON(`${CONV_API}/${id}/`, {
            method: 'PATCH',
            body: JSON.stringify({ title }),
        });
        const idx = conversations.findIndex(c => c.id === id);
        if (idx >= 0) {
            conversations[idx].title = updated.title || title;
            conversations[idx].updated_at = updated.updated_at || conversations[idx].updated_at;
            conversations[idx].updatedAt = conversations[idx].updated_at;
        }
    } catch (e) {
        console.error('Falha ao renomear conversa:', e);
    }
    renderHistoryList();
};

function updateHistoryCountInfo() {
    const el = document.getElementById('historyCountInfo');
    if (!el) return;
    const vis = visibleConversations();
    const n = vis.length;
    const msgs = vis.reduce((a, c) => a + getConvMsgCount(c), 0);
    el.textContent = n === 0 ? 'Nenhuma conversa salva.' : `${n} conversa(s) · ${msgs} mensagem(ns) no servidor.`;
}

window.clearAllHistory = async function() {
    const btn = document.getElementById('clearAllHistoryBtn');
    const feedback = document.getElementById('clearHistoryFeedback');
    // Confirmação em 2 etapas: primeiro clique arma, segundo confirma
    if (btn && !btn.dataset.armed) {
        btn.dataset.armed = '1';
        const span = btn.querySelector('span');
        if (span) span.textContent = 'Clique novamente para confirmar — NÃO pode ser desfeito!';
        btn.style.background = 'rgba(219, 68, 85, 0.18)';
        if (feedback) {
            feedback.style.display = 'block';
            feedback.style.color = '#db4455';
            feedback.textContent = 'Tem certeza? Esta ação apaga tudo no servidor e NÃO pode ser desfeita. Clique novamente para confirmar.';
        }
        setTimeout(() => {
            if (!btn) return;
            delete btn.dataset.armed;
            const s = btn.querySelector('span');
            if (s) s.textContent = 'Excluir todo o histórico';
            btn.style.background = '';
            if (feedback) feedback.style.display = 'none';
        }, 5000);
        return;
    }
    if (btn) delete btn.dataset.armed;
    if (isWaiting && currentAbortController) { try { currentAbortController.abort(); } catch (e) {} }
    try {
        await apiJSON(CONV_API + '/clear/', { method: 'POST', body: JSON.stringify({}) });
        conversations = [];
        currentConversationId = null;
    } catch (e) {
        console.error('Falha ao limpar histórico na API:', e);
        if (feedback) {
            feedback.style.display = 'block';
            feedback.style.color = '#db4455';
            feedback.textContent = 'Falha ao excluir no servidor. Tente novamente.';
            setTimeout(() => { feedback.style.display = 'none'; }, 3000);
        }
        return;
    }
    syncChatHistory();
    renderHistoryList();
    renderCurrentMessages();
    updateHistoryCountInfo();
    if (btn) {
        const span = btn.querySelector('span');
        if (span) span.textContent = 'Excluir todo o histórico';
        btn.style.background = '';
    }
    if (feedback) {
        feedback.style.display = 'block';
        feedback.style.color = '#22c55e';
        feedback.textContent = 'Histórico apagado com sucesso.';
        setTimeout(() => { feedback.style.display = 'none'; }, 3000);
    }
};

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
    const hasContent = promptInput.value.trim().length > 0 || !!currentImageBase64 || pendingDocs.length > 0;
    sendBtn.classList.toggle('active', hasContent);
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
    // esconde o hero na primeira mensagem da conversa atual
    if (welcomeHero.style.display === 'none') return;
    firstMessage = false;
    welcomeHero.classList.add('dismissing');
    welcomeHero.style.display = 'none';
    chatArea.style.display = 'block';
}

// marked config (seguro se CDN falhar)
try {
    if (window.marked && window.hljs) {
        marked.setOptions({
            highlight: function (code, lang) {
                try {
                    const language = hljs.getLanguage(lang) ? lang : 'plaintext';
                    return hljs.highlight(code, { language }).value;
                } catch (e) { return code; }
            }
        });
    }
} catch (e) { console.warn('marked init falhou:', e); }

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

// ── Anexos: 1 imagem p/ visão + até 6 documentos (PDF, TXT, MD, CSV, JSON, DOCX) ──
const DOC_EXTS = ['pdf', 'txt', 'md', 'markdown', 'csv', 'json', 'log', 'docx'];
const MAX_DOCS = 6;
const MAX_FILE_BYTES = 10 * 1024 * 1024;
let pendingDocs = []; // [{filename, mime, content_base64}]

function refreshPreviewStrip() {
    const hasImg = !!currentImageBase64;
    const hasDocs = pendingDocs.length > 0;
    previewStrip.style.display = (hasImg || hasDocs) ? 'flex' : 'none';
    inputBox.classList.toggle('has-image', hasImg);
    renderDocChips();
    if (!isWaiting) {
        const hasText = promptInput.value.trim().length > 0;
        sendBtn.classList.toggle('active', hasText || hasImg || hasDocs);
    }
}

function renderDocChips() {
    let box = document.getElementById('docChips');
    if (!box && pendingDocs.length) {
        box = document.createElement('div');
        box.id = 'docChips';
        box.className = 'doc-chips';
        previewStrip.appendChild(box);
    }
    if (!box) return;
    box.innerHTML = '';
    pendingDocs.forEach((d, i) => {
        const chip = document.createElement('div');
        chip.className = 'doc-chip';
        const name = document.createElement('span');
        name.className = 'doc-chip-name';
        name.textContent = d.filename;
        const x = document.createElement('button');
        x.className = 'doc-chip-remove';
        x.title = 'Remover anexo';
        x.textContent = '✕';
        x.onclick = () => window.removeDoc(i);
        chip.appendChild(name);
        chip.appendChild(x);
        box.appendChild(chip);
    });
    if (!pendingDocs.length && box) box.remove();
}

function attachmentNote() {
    if (!pendingDocs.length) return '';
    return '\n📎 Anexos: ' + pendingDocs.map(d => d.filename).join(', ');
}

function clearAttachments() {
    pendingDocs = [];
    currentImageBase64 = null;
    const inp = document.getElementById('imageInput');
    if (inp) inp.value = '';
    const img = document.getElementById('imagePreview');
    if (img) img.src = '';
    refreshPreviewStrip();
}

window.handleImageUpload = function(event) {
    const files = Array.from(event.target.files || []);
    if (!files.length) return;
    files.forEach(file => {
        if (file.size > MAX_FILE_BYTES) {
            alert(`"${file.name}" excede 10MB e foi ignorado.`);
            return;
        }
        const isImage = (file.type || '').startsWith('image/');
        const ext = (file.name.split('.').pop() || '').toLowerCase();
        const reader = new FileReader();
        reader.onload = e => {
            const base64 = String(e.target.result).split(',')[1] || '';
            if (isImage && !currentImageBase64) {
                // primeira imagem vai p/ visão do modelo
                currentImageBase64 = base64;
                document.getElementById('imagePreview').src = e.target.result;
            } else if (!isImage && DOC_EXTS.includes(ext)) {
                if (pendingDocs.length >= MAX_DOCS) {
                    alert(`Máximo de ${MAX_DOCS} documentos por mensagem.`);
                    return;
                }
                if (!pendingDocs.some(d => d.filename === file.name)) {
                    pendingDocs.push({ filename: file.name, mime: file.type || 'application/octet-stream', content_base64: base64 });
                }
            } else if (isImage) {
                alert('Apenas 1 imagem por mensagem (visão do modelo).');
                return;
            } else {
                alert(`Tipo não suportado: "${file.name}". Use imagem, PDF, TXT, MD, CSV, JSON ou DOCX.`);
                return;
            }
            refreshPreviewStrip();
        };
        reader.readAsDataURL(file);
    });
    event.target.value = '';
};

window.removeDoc = function(index) {
    pendingDocs.splice(index, 1);
    refreshPreviewStrip();
};

window.removeImage = function() {
    currentImageBase64 = null;
    const inp = document.getElementById('imageInput');
    if (inp) inp.value = '';
    const img = document.getElementById('imagePreview');
    if (img) img.src = '';
    refreshPreviewStrip();
    if (!isWaiting && !promptInput.value.trim() && !pendingDocs.length) sendBtn.classList.remove('active');
};

promptInput.addEventListener('input', function () {
    this.style.height = '40px';
    this.style.height = Math.min(this.scrollHeight, 140) + 'px';
    if (!isWaiting) {
        sendBtn.classList.toggle('active', this.value.trim().length > 0 || !!currentImageBase64 || pendingDocs.length > 0);
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
    const cleanText = (text || '').replace(/\[METRICS\].*?\[\/METRICS\]/g, '');
    const isThinkingNow = isStreaming; // Durante o streaming, assumimos que está pensando se a tag estiver aberta
    const md = (t) => {
        try {
            if (window.marked && marked.parse) return marked.parse(t || '');
        } catch (e) {}
        return (t || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\n/g, '<br>');
    };

    // 1. Tags <think>
    const thinkMatch = cleanText.match(/<think>([\s\S]*?)(?:<\/think>|$)/);
    if (thinkMatch) {
        const parts = cleanText.split(/<think>|<\/think>/);
        const isFinal = cleanText.includes('</think>');
        return md(parts[0] || '') +
            thinkBlockHTML(thinkMatch[1].trim(), 'Processo de Raciocínio', !isFinal) +
            md(parts[2] || '');
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

        return md(preText) +
            thinkBlockHTML(thinkingPart.trim(), 'Processo de Raciocínio', isStreaming) +
            md(finalAnswer);
    }

    return md(cleanText);
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
        <div class="msg-body">
            <div class="msg-bubble">${imgTag}${(text || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\n/g, '<br>')}</div>
            <div class="msg-meta msg-meta-user">${time}</div>
        </div>`;
    } else {
        wrap.innerHTML = `
        <div class="avatar bot">✦</div>
        <div class="msg-body">
            <div class="msg-meta">
                <span class="msg-name">AgroMind</span>
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

async function finalizeBotMessage(el) {
    const rawText = (el.dataset.rawText || '').trim();
    if (!rawText) {
        // stream vazio/erro/abort antes de qualquer token: não salva lixo no servidor,
        // só informa na tela
        try {
            el.querySelector('.msg-bubble').innerHTML =
                renderBotContent('*Sem resposta do modelo. Tente enviar novamente.*', false);
        } catch (e) {}
        return;
    }

    // Renderiza uma última vez com isStreaming = false para fechar o card e separar a resposta
    el.querySelector('.msg-bubble').innerHTML = renderBotContent(rawText, false);

    await saveAssistantMessage(rawText);
}

function showLoader() {
    const wrap = document.createElement('div');
    wrap.className = 'message-wrap bot';
    wrap.id = 'loaderMsg';
    wrap.innerHTML = `
    <div class="avatar bot">✦</div>
    <div class="msg-body">
        <div class="msg-meta"><span class="msg-name">AgroMind</span></div>
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

window.updateRAM = async function() {
    try {
        const response = await fetch('/api/status/');
        const data = await response.json();
        const ramDisplay = document.getElementById('ramDisplay');
        if (ramDisplay) {
            ramDisplay.innerText = `${data.process_ram_mb} MB`;
        }
        // Opcional: atualizar indicador de GPU se existir no UI
        const gpuStatus = document.getElementById('gpuStatus');
        if (gpuStatus) {
            gpuStatus.innerText = data.gpu_enabled ? 'GPU Ativa' : 'CPU';
        }
    } catch (err) {
        console.error("Erro ao atualizar status do sistema:", err);
    }
};

// Substituir chamadas internas de updateRAM() para window.updateRAM()
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
            window.updateRAM(); // Atualiza uso de memória
        } else {
            alert("Erro ao trocar o modelo.");
            loadModels();
        }
    } catch (err) {
        console.error("Erro:", err);
        loadModels();
    }
}
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
            window.updateRAM(); // Atualiza uso de memória
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
            window.updateRAM(); // Atualiza uso de memória

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

// ── Modelo Remoto ──
async function saveRemoteConfig() {
    const config = {
        enabled: document.getElementById('remoteToggle').checked,
        api_url: document.getElementById('remoteApiUrl').value.trim(),
        model: document.getElementById('remoteModel').value.trim(),
    };

    const btn = document.getElementById('remoteSaveBtn');
    const originalText = btn.querySelector('span').innerText;
    btn.querySelector('span').innerText = 'Salvando...';
    btn.disabled = true;

    try {
        const resp = await fetch('/api/remote-config/save/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ config })
        });
        if (resp.ok) {
            localStorage.setItem('remoteConfig', JSON.stringify(config));
            btn.querySelector('span').innerText = 'Salvo!';
            btn.style.borderColor = 'var(--primary)';
            setTimeout(() => {
                btn.querySelector('span').innerText = originalText;
                btn.disabled = false;
                btn.style.borderColor = '';
            }, 2000);
        } else {
            alert('Erro ao salvar configuração.');
            btn.querySelector('span').innerText = originalText;
            btn.disabled = false;
        }
    } catch (err) {
        localStorage.setItem('remoteConfig', JSON.stringify(config));
        btn.querySelector('span').innerText = 'Salvo (local)';
        setTimeout(() => {
            btn.querySelector('span').innerText = originalText;
            btn.disabled = false;
        }, 2000);
    }
}

async function testRemoteConnection() {
    const config = {
        enabled: document.getElementById('remoteToggle').checked,
        api_url: document.getElementById('remoteApiUrl').value.trim(),
        model: document.getElementById('remoteModel').value.trim(),
    };

    const statusEl = document.getElementById('remoteStatus');
    statusEl.style.display = 'block';
    statusEl.innerHTML = '<span style="color: var(--text-dim);">Testando conexão...</span>';

    try {
        const resp = await fetch('/api/remote-config/test/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ config })
        });
        const data = await resp.json();
        if (data.success) {
            const ver = data.version ? ` (v${data.version})` : '';
            statusEl.innerHTML = `<span style="color: #4caf50;">✓ OpenCode conectado${ver}!</span>`;
            statusEl.style.background = 'rgba(76, 175, 80, 0.08)';
            statusEl.style.border = '1px solid rgba(76, 175, 80, 0.2)';
        } else {
            statusEl.innerHTML = `<span style="color: #db4455;">✗ Erro: ${data.error}</span>`;
            statusEl.style.background = 'rgba(219, 68, 85, 0.08)';
            statusEl.style.border = '1px solid rgba(219, 68, 85, 0.2)';
        }
    } catch (err) {
        statusEl.innerHTML = `<span style="color: #db4455;">✗ Erro de conexão: ${err.message}</span>`;
        statusEl.style.background = 'rgba(219, 68, 85, 0.08)';
        statusEl.style.border = '1px solid rgba(219, 68, 85, 0.2)';
    }
}

function loadRemoteConfig() {
    function apply(cfg) {
        if (!cfg || !cfg.api_url) return;
        document.getElementById('remoteToggle').checked = cfg.enabled || false;
        document.getElementById('remoteApiUrl').value = cfg.api_url;
        document.getElementById('remoteModel').value = cfg.model || '';
    }
    fetch('/api/remote-config/load/')
        .then(r => r.json())
        .then(config => { apply(config); if (!config?.api_url) { const s = localStorage.getItem('remoteConfig'); if (s) { try { apply(JSON.parse(s)); } catch (e) {} } } })
        .catch(() => { const s = localStorage.getItem('remoteConfig'); if (s) { try { apply(JSON.parse(s)); } catch (e) {} } });
}

window.sendMessage = async function() {
    if (isWaiting) {
        if (currentAbortController) {
            currentAbortController.abort();
        }
        return;
    }

    const text = promptInput.value.trim();
    if (!text && !currentImageBase64 && !pendingDocs.length) return;
    dismissWelcome();

    const temperature = parseFloat(document.getElementById('temperature')?.value) || 0.1;

    const sysPromptEl = document.getElementById('systemPrompt');
    const systemPrompt = sysPromptEl ? sysPromptEl.value : null;

    const ragToggleEl = document.getElementById('ragToggle');
    const useRag = ragToggleEl ? ragToggleEl.checked : true;

    // Remote config
    const remoteToggleEl = document.getElementById('remoteToggle');
    const useRemote = remoteToggleEl ? remoteToggleEl.checked : false;
    let remoteConfig = {};
    if (useRemote) {
        remoteConfig = {
            enabled: true,
            api_url: document.getElementById('remoteApiUrl').value.trim(),
            model: document.getElementById('remoteModel').value.trim(),
        };
    }

    const payloadImage = currentImageBase64;
    const payloadDocs = pendingDocs.map(d => ({ ...d }));
    const note = attachmentNote();
    const userText = (text || (payloadImage ? '[Imagem enviada]' : '[Documentos enviados]')) + note;

    // rascunho (tela inicial): cria no servidor só agora, no primeiro envio
    try {
        await ensureConversationForSend(userText);
    } catch (e) {
        console.error('Falha ao criar conversa na API:', e);
        return;
    }

    await saveUserMessage(userText, payloadImage);

    chatArea.appendChild(createMsg(userText, true, payloadImage));
    promptInput.value = '';
    promptInput.style.height = '40px';
    clearAttachments();

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
                prompt: text || (payloadImage ? "Analise esta imagem." : (payloadDocs.length ? "Analise os documentos anexados." : "")),
                temperature: temperature,
                image_base64: payloadImage,
                attachments: payloadDocs,
                history: chatHistory.slice(0, -1),
                system_prompt: systemPrompt,
                use_rag: useRag,
                use_remote: useRemote,
                remote_config: remoteConfig
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
        let streamError = null;

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (!line.trim()) continue;
                try {
                    const packet = JSON.parse(line);
                    const event = packet.event;
                    const data = packet.data;
                    let bubble = botEl.querySelector('.msg-bubble');

                    if (event === 'thought') {
                        let thinkCard = bubble.querySelector('.think-block');
                        if (!thinkCard) {
                            bubble.insertAdjacentHTML('afterbegin', thinkBlockHTML('', 'Processo de Raciocínio', true));
                            thinkCard = bubble.querySelector('.think-block');
                        }
                        thinkCard.querySelector('.think-scroll').innerText += data;
                    } else if (event === 'message') {
                        let thinkCard = bubble.querySelector('.think-block');
                        if (thinkCard) thinkCard.classList.remove('expanded');

                        botEl.dataset.rawText += data;
                        bubble.innerHTML = renderBotContent(botEl.dataset.rawText, true);
                    } else if (event === 'error') {
                        streamError = data;
                        console.error('Erro no stream:', data);
                        bubble.innerHTML = renderBotContent(`**Erro do modelo:** ${data}`, false);
                    } else if (event === 'metrics') {
                        let badgeContainer = document.createElement('div');
                        badgeContainer.innerHTML = `<div class="perf-badge">
                            <span><b>${data.tps.toFixed(2)}</b> t/s</span>
                            <div class="perf-sep"></div>
                            <span><b>${data.tokens}</b> tokens</span>
                            <div class="perf-sep"></div>
                            <span><b>${data.duration.toFixed(2)}s</b></span>
                        </div>`;
                        botEl.querySelector('.msg-body').appendChild(badgeContainer.firstChild);
                    } else if (event === 'done') {
                        console.log('Stream concluído com sucesso.');
                    }
                } catch (e) {
                    console.error('Erro ao processar JSON:', e, 'Linha:', line);
                }
            }
        }
        if (streamError && !(botEl.dataset.rawText || '').trim()) {
            // erro puro (ex: modelo não carregado): mostra na tela mas NÃO salva
            // resposta vazia no servidor, senão o reload mostraria bolha vazia
            removeLoader();
        } else {
            try { await finalizeBotMessage(botEl); }
            catch (e) { console.error('Falha ao persistir resposta na API:', e); }
        }
    } catch (err) {
        if (err.name === 'AbortError') {
            console.log('Stream aborted');
            if (botEl) {
                try { await finalizeBotMessage(botEl); }
                catch (e) { console.error('Falha ao persistir resposta interrompida:', e); }
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

window.copyCommand = async function() {
    const commandEl = document.getElementById('opencodeCommand');
    const btn = document.querySelector('.copy-command-btn');
    const span = btn.querySelector('span');

    if (!commandEl || !btn) return;

    try {
        await navigator.clipboard.writeText(commandEl.textContent);
        btn.classList.add('copied');
        span.innerText = 'Copiado!';
        btn.querySelector('svg').innerHTML = '<polyline points="20 6 9 17 4 12"/>';

        setTimeout(() => {
            btn.classList.remove('copied');
            span.innerText = 'Copiar';
            btn.querySelector('svg').innerHTML = '<rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>';
        }, 2000);
    } catch (err) {
        console.error('Erro ao copiar comando:', err);
    }
};

// ── Sidebar mobile (hamburger) ──
window.toggleSidebar = function(force) {
    const open = typeof force === 'boolean' ? force : !document.body.classList.contains('sidebar-open');
    document.body.classList.toggle('sidebar-open', open);
};

function closeSidebarOnMobile() {
    try {
        if (window.innerWidth <= 900) document.body.classList.remove('sidebar-open');
    } catch (e) {}
}

// Inicializa a interface (histórico vem da API)
(async function initHistory() {
    try { await loadStore(); } catch (e) { console.warn('loadStore falhou:', e); }
})();
// Liga botão Nova conversa também via JS (não depende só do onclick inline)
try {
    const nb = document.getElementById('newChatBtn');
    if (nb && !nb.dataset.bound) {
        nb.dataset.bound = '1';
        nb.addEventListener('click', () => window.newConversation());
    }
} catch (e) {}
loadModels();
loadRemoteConfig();
window.updateRAM();
setInterval(window.updateRAM, 5000);
