/* ── Modo Live (voz Piper) — módulo isolado ──────────────────────────
 * Não edita script.js. Envolve window.sendMessage: depois que o turno
 * termina, fala a resposta final via /api/live/speak, frase por frase.
 * Com Live desligado, comportamento é 100% o original.
 * ──────────────────────────────────────────────────────────────────── */
(function () {
    'use strict';

    const API = {
        health: '/api/live/health/',
        voices: '/api/live/voices/',
        speak: '/api/live/speak/',
        split: '/api/live/split/',
    };

    const state = {
        enabled: false,
        engineOk: null,   // null = não verificado, true/false após /health
        speaking: false,
        stopFlag: false,
        voice: null,
        speed: 0.9, // levemente mais devagar = mais natural
        audio: null,
        runId: 0,
    };

    // Com Live ligado, o modelo responde curto (fala mais rápido).
    // Injetado só no corpo da requisição — não altera o campo Diretriz na tela.
    // Dobradinha: instrução curta + teto físico de tokens (modelo pequeno ignora estilo).
    const LIVE_BRIEF = '[MODO LIVE: seja objetivo e direto. '
        + 'Responda em no máximo 4 frases curtas. '
        + 'PROIBIDO listas, código, saudações longas e perguntas de volta.]';
    const LIVE_MAX_TOKENS = 256;

    // Intercepta só /api/chat/ e /api/chat-stream/ p/ anexar a brevidade.
    // Com Live desligado, o fetch passa 100% intacto.
    function wrapFetch() {
        try {
            if (window.fetch.__liveWrapped) return;
            const origFetch = window.fetch.bind(window);
            window.fetch = function (input, init) {
                try {
                    const url = typeof input === 'string' ? input : (input && input.url) || '';
                    const isChat = url.includes('/api/chat-stream/') || url.endsWith('/api/chat/');
                    const body = init && typeof init.body === 'string' ? init.body : null;
                    if (state.enabled && isChat && body) {
                        const payload = JSON.parse(body);
                        if (payload && typeof payload.prompt === 'string') {
                            const cur = payload.system_prompt || '';
                            if (!cur.includes('MODO LIVE')) {
                                payload.system_prompt = (cur ? cur + '\n\n' : '') + LIVE_BRIEF;
                            }
                            payload.max_tokens = LIVE_MAX_TOKENS;
                            init = { ...init, body: JSON.stringify(payload) };
                        }
                    }
                } catch (e) { /* segue sem alterar */ }
                return origFetch(input, init);
            };
            window.fetch.__liveWrapped = true;
        } catch (e) { console.warn('live fetch wrap falhou:', e); }
    }
    wrapFetch();

    function toast(msg, isErr) {
        try {
            let el = document.getElementById('liveToast');
            if (!el) {
                el = document.createElement('div');
                el.id = 'liveToast';
                el.className = 'live-toast';
                document.body.appendChild(el);
            }
            el.textContent = msg;
            el.classList.toggle('err', !!isErr);
            el.classList.add('show');
            clearTimeout(el._t);
            el._t = setTimeout(() => el.classList.remove('show'), 3500);
        } catch (e) { console.warn('live toast:', e); }
    }

    function paintBtn() {
        const btn = document.getElementById('liveBtn');
        if (!btn) return;
        btn.classList.toggle('live-on', state.enabled);
        btn.classList.toggle('speaking', state.speaking);
        btn.title = state.enabled
            ? (state.speaking ? 'Live ligado — falando… (clique p/ desligar)' : 'Live ligado — clique p/ desligar')
            : 'Modo Live: modelo fala a resposta (Piper, local)';
        const label = btn.querySelector('.live-dot');
        if (label) label.style.opacity = state.enabled ? '1' : '0';
    }

    function stopAudio() {
        state.stopFlag = true;
        state.runId += 1;
        try {
            if (state.audio) {
                state.audio.pause();
                state.audio.src = '';
                state.audio = null;
            }
        } catch (e) {}
        state.speaking = false;
        paintBtn();
    }

    function localSplit(text, limit) {
        limit = limit || 220;
        const clean = String(text || '')
            .replace(/```[\s\S]*?```/g, ' ')
            .replace(/`([^`]*)`/g, '$1')
            .replace(/!\[([^\]]*)\]\([^)]*\)/g, '$1')
            .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
            .replace(/^#{1,6}\s+/gm, '')
            .replace(/(\*\*|__)(.*?)\1/g, '$2')
            .replace(/https?:\/\/\S+/g, ' ')
            .replace(/\|/g, ', ')
            .replace(/\s+/g, ' ').trim();
        if (!clean) return [];
        const parts = clean.split(/(?<=[.!?…])\s+|\n+/);
        const out = [];
        let buf = '';
        for (const p of parts) {
            const c = (buf ? buf + ' ' : '') + p.trim();
            if (c.length <= limit) buf = c;
            else {
                if (buf) out.push(buf);
                buf = p.trim().slice(0, limit);
            }
        }
        if (buf) out.push(buf);
        return out;
    }

    async function splitText(text) {
        try {
            const res = await fetch(API.split, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text, limit: 220 }),
            });
            if (res.ok) {
                const data = await res.json();
                if (Array.isArray(data.chunks) && data.chunks.length) return data.chunks;
            }
        } catch (e) { /* cai pro split local */ }
        return localSplit(text);
    }

    function playBlob(url, runId) {
        return new Promise((resolve) => {
            const audio = new Audio(url);
            state.audio = audio;
            audio.onended = () => resolve(true);
            audio.onerror = () => resolve(false);
            try {
                const p = audio.play();
                if (p && p.catch) p.catch(() => resolve('blocked'));
            } catch (e) { resolve('blocked'); return; }
            // permite interromper entre frases
            const iv = setInterval(() => {
                if (runId !== state.runId || state.stopFlag) {
                    clearInterval(iv);
                    try { audio.pause(); } catch (e) {}
                    resolve(false);
                }
            }, 120);
            audio.onended = () => { clearInterval(iv); resolve(true); };
        });
    }

    async function speakChunks(chunks, runId) {
        let ok = 0, fail = 0, blocked = false;
        for (const chunk of chunks) {
            if (runId !== state.runId || state.stopFlag || !state.enabled) return;
            let url = null;
            try {
                const res = await fetch(API.speak, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: chunk, voice: state.voice, speed: state.speed }),
                });
                if (res.status === 503) {
                    const data = await res.json().catch(() => ({}));
                    toast(data.error || 'Voz indisponível. Instale: pip install piper-tts onnxruntime', true);
                    return;
                }
                if (!res.ok) {
                    fail += 1;
                    const detail = await res.text().catch(() => '');
                    console.warn('live speak pulou frase:', res.status, detail.slice(0, 200));
                    if (fail === 1) toast(`Falha ao gerar voz (HTTP ${res.status}). Verifique o terminal do servidor.`, true);
                    continue;
                }
                const blob = await res.blob();
                if (!blob.size) { fail += 1; continue; }
                url = URL.createObjectURL(blob);
                state.speaking = true;
                paintBtn();
                const r = await playBlob(url, runId);
                if (r === 'blocked') { blocked = true; break; }
                if (r) ok += 1; else fail += 1;
            } catch (e) {
                fail += 1;
                console.warn('live speak erro:', e);
            } finally {
                if (url) setTimeout(() => URL.revokeObjectURL(url), 10000);
            }
        }
        if (blocked) toast('Navegador bloqueou o áudio. Clique na página e envie outra mensagem.', true);
        else if (!ok && fail) toast('Não consegui gerar nem tocar a voz. Veja o console (F12) e o terminal.', true);
    }

    function lastBotText() {
        try {
            const area = document.getElementById('chatArea');
            if (!area) return '';
            // classe real do script.js: "message-wrap bot" (user: "message-wrap user")
            let msgs = area.querySelectorAll('.message-wrap.bot');
            if (!msgs.length) msgs = area.querySelectorAll('.message-wrap:not(.user)');
            if (!msgs.length) return '';
            const last = msgs[msgs.length - 1];
            // script.js guarda o texto cru aqui (sem HTML) — ideal p/ TTS
            if (last.dataset && last.dataset.rawText) return last.dataset.rawText;
            const bubble = last.querySelector('.msg-bubble');
            return bubble ? bubble.innerText : last.innerText;
        } catch (e) { return ''; }
    }

    async function onTurnFinished() {
        if (!state.enabled) return;
        const runId = state.runId;
        const text = lastBotText();
        if (!text || !text.trim()) {
            toast('Live: não encontrei texto na resposta para falar.', true);
            return;
        }
        state.stopFlag = false;
        const chunks = await splitText(text);
        if (!chunks.length || runId !== state.runId || !state.enabled) return;
        toast(`Falando resposta (${chunks.length} trecho${chunks.length > 1 ? 's' : ''})…`);
        await speakChunks(chunks.slice(0, 12), runId); // teto: 12 frases por resposta
        state.speaking = false;
        paintBtn();
    }

    async function checkEngine() {
        try {
            const res = await fetch(API.health);
            if (!res.ok) return false;
            const data = await res.json();
            return !!(data.live && data.live.available);
        } catch (e) { return false; }
    }

    async function toggle() {
        if (state.enabled) {
            state.enabled = false;
            stopAudio();
            paintBtn();
            toast('Modo Live desligado.');
            return;
        }
        // ligando: verifica motor primeiro (primeira voz pode baixar ~60MB)
        toast('Ativando Live… verificando voz local.');
        const ok = await checkEngine();
        state.engineOk = ok;
        if (!ok) {
            toast('Voz indisponível. Rode: pip install piper-tts onnxruntime (+ espeak-ng no Windows). O chat segue normal.', true);
            return;
        }
        state.enabled = true;
        state.stopFlag = false;
        state.runId += 1;
        paintBtn();
        try {
            const res = await fetch(API.voices);
            if (res.ok) {
                const data = await res.json();
                if (data.default && !state.voice) state.voice = data.default;
            }
        } catch (e) {}
        toast('Live ligado — respostas breves e faladas.');
    }

    // Envolve o sendMessage original SEM alterá-lo: nova mensagem = interrompe fala atual.
    function wrapSendMessage() {
        try {
            if (typeof window.sendMessage !== 'function' || window.sendMessage.__liveWrapped) return;
            const orig = window.sendMessage;
            const wrapped = async function () {
                stopAudio(); // interrompe fala anterior ao enviar
                state.stopFlag = false;
                const myRun = state.runId;
                try {
                    return await orig.apply(this, arguments);
                } finally {
                    if (state.enabled && myRun === state.runId) onTurnFinished();
                    else { state.speaking = false; paintBtn(); }
                }
            };
            wrapped.__liveWrapped = true;
            window.sendMessage = wrapped;
        } catch (e) { console.warn('live wrap falhou:', e); }
    }

    // script.js define window.sendMessage no fim do arquivo; tenta envolver com retry.
    let tries = 0;
    const iv = setInterval(() => {
        tries += 1;
        if (typeof window.sendMessage === 'function') {
            wrapSendMessage();
            clearInterval(iv);
        } else if (tries > 50) clearInterval(iv);
    }, 200);

    window.LiveMode = {
        toggle,
        stop: stopAudio,
        get enabled() { return state.enabled; },
        setVoice(v) { state.voice = v; },
        setSpeed(s) { state.speed = Math.max(0.5, Math.min(2.0, Number(s) || 1.0)); },
    };

    document.addEventListener('DOMContentLoaded', paintBtn);
})();
