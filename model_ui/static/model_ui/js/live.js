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
        transcribe: '/api/live/transcribe/',
    };
    const REC_MAX_S = 60;

    const state = {
        enabled: false,
        engineOk: null,   // null = não verificado, true/false após /health
        speaking: false,
        stopFlag: false,
        voice: null,
        speed: 0.8, // cadência calma e natural
        audio: null,
        runId: 0,
        keepAlive: { active: false, restarts: 0, since: 0 },
    };
    const KEEP_MAX_RESTARTS = 24; // ~3 min de espera no total
    const KEEP_MAX_S = 180;

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
        if (btn) {
            btn.classList.toggle('live-on', state.enabled);
            btn.classList.toggle('speaking', state.speaking);
            btn.title = state.enabled
                ? (state.speaking ? 'Live ligado — falando… (clique p/ desligar)' : 'Live ligado — clique p/ desligar')
                : 'Modo Live: conversa por voz (local)';
            const label = btn.querySelector('.live-dot');
            if (label) label.style.opacity = state.enabled ? '1' : '0';
        }
        // status do painel live
        try {
            const listening = liveIsListening();
            const st = document.getElementById('liveStatus');
            if (st) {
                st.textContent = state.speaking ? 'Falando…'
                    : listening ? 'Ouvindo…'
                    : state.busy ? 'Pensando…'
                    : 'Pode falar';
            }
            const orb = document.getElementById('liveOrb');
            if (orb) {
                orb.classList.toggle('speaking', state.speaking);
                orb.classList.toggle('listening', listening && !state.speaking);
                orb.classList.toggle('thinking', state.busy && !state.speaking);
            }
        } catch (e) {}
    }

    // ── Painel Live: substitui a caixa de texto quando ativo ──
    function buildLivePanel() {
        try {
            if (document.getElementById('livePanel')) return;
            const zone = document.querySelector('.input-zone');
            if (!zone) return;
            const panel = document.createElement('div');
            panel.id = 'livePanel';
            panel.style.display = 'none';
            panel.innerHTML =
                '<div class="live-orb" id="liveOrb"><canvas id="liveWave" width="280" height="72"></canvas></div>' +
                '<div class="live-status" id="liveStatus">Pode falar</div>' +
                '<button class="live-end" id="liveEndBtn" title="Encerrar Live">✕ Encerrar live</button>';
            zone.parentNode.insertBefore(panel, zone.nextSibling);
            document.getElementById('liveEndBtn').addEventListener('click', () => toggle());
            startWaveLoop();
        } catch (e) { console.warn('live panel falhou:', e); }
    }

    function showLivePanel(on) {
        try {
            buildLivePanel();
            const zone = document.querySelector('.input-zone');
            const panel = document.getElementById('livePanel');
            if (zone) zone.style.display = on ? 'none' : '';
            if (panel) panel.style.display = on ? 'flex' : 'none';
        } catch (e) {}
    }

    // ── Web Audio: waveform reage ao mic e à fala ──
    function liveEnsureCtx() {
        try {
            if (!state.actx) {
                const AC = window.AudioContext || window.webkitAudioContext;
                if (!AC) return null;
                state.actx = new AC();
                state.micAn = state.actx.createAnalyser();
                state.micAn.fftSize = 256;
                state.ttsAn = state.actx.createAnalyser();
                state.ttsAn.fftSize = 256;
            }
            if (state.actx.state === 'suspended') state.actx.resume().catch(() => {});
            return state.actx;
        } catch (e) { return null; }
    }

    async function liveMicTap() {
        try {
            const ctx = liveEnsureCtx();
            if (!ctx || state.micStream) return;
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            state.micStream = stream;
            ctx.createMediaStreamSource(stream).connect(state.micAn);
        } catch (e) { /* sem visual, painel segue normal */ }
    }

    function liveMicUntap() {
        try {
            if (state.micStream) state.micStream.getTracks().forEach(t => t.stop());
        } catch (e) {}
        state.micStream = null;
    }

    function liveTapTTS(audioEl) {
        try {
            const ctx = liveEnsureCtx();
            if (!ctx) return;
            ctx.createMediaElementSource(audioEl).connect(state.ttsAn);
            state.ttsAn.connect(ctx.destination);
        } catch (e) {}
    }

    // ouvindo = ditado do Chrome ativo (classe do botão) ou gravação local
    function liveIsListening() {
        try {
            const b = document.getElementById('micBtn');
            if (b && b.classList.contains('recording')) return true;
        } catch (e) {}
        return !!state.recording;
    }

    function waveLevels(n) {
        const out = new Array(n).fill(0);
        try {
            let an = null;
            if (state.speaking && state.ttsAn) an = state.ttsAn;
            else if (liveIsListening() && state.micAn && state.micStream) an = state.micAn;
            if (!an) return { bars: out, live: false };
            const buf = new Uint8Array(an.frequencyBinCount);
            an.getByteFrequencyData(buf);
            for (let i = 0; i < n; i++) {
                const idx = Math.floor(Math.pow(i / n, 1.4) * buf.length * 0.7);
                out[i] = buf[idx] / 255;
            }
            return { bars: out, live: true };
        } catch (e) { return { bars: out, live: false }; }
    }

    let waveT = 0;
    function startWaveLoop() {
        try {
            const cv = document.getElementById('liveWave');
            if (!cv) return;
            const cx = cv.getContext('2d');
            const N = 48;
            function frame() {
                try {
                    const panel = document.getElementById('livePanel');
                    if (panel && panel.style.display !== 'none') {
                        const W = cv.width, H = cv.height;
                        cx.clearRect(0, 0, W, H);
                        const { bars, live } = waveLevels(N);
                        waveT += 0.09;
                        const gap = 2, bw = (W - (N - 1) * gap) / N;
                        cx.fillStyle = state.speaking ? '#22c55e'
                            : liveIsListening() ? '#4ade80'
                            : 'rgba(130,130,150,.55)';
                        for (let i = 0; i < N; i++) {
                            let v;
                            if (live) v = bars[i];
                            else if (state.busy && !state.speaking) v = 0.16 + 0.12 * Math.sin(waveT * 1.6 + i * 0.3);
                            else v = 0.07 + 0.05 * Math.sin(waveT + i * 0.5);
                            const h = Math.max(3, v * (H - 8));
                            const x = i * (bw + gap), y = (H - h) / 2;
                            if (cx.roundRect) { cx.beginPath(); cx.roundRect(x, y, bw, h, bw / 2); cx.fill(); }
                            else cx.fillRect(x, y, bw, h);
                        }
                    }
                } catch (e) {}
                requestAnimationFrame(frame);
            }
            requestAnimationFrame(frame);
        } catch (e) {}
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
            liveTapTTS(audio); // waveform reage à fala
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

    // Fecha o loop do bate-papo: após falar, religa o mic sozinho
    // (só se o turno veio de voz — quem digitou não é interrompido).
    function restartMicForNextTurn(runId) {
        try {
            if (!state.enabled || runId !== state.runId) return;
            const btn = document.getElementById('micBtn');
            if (!btn || btn.classList.contains('recording')) return;
            if (typeof window.toggleVoice !== 'function') return;
            setTimeout(() => {
                try {
                    if (!state.enabled || runId !== state.runId || state.busy) return;
                    const b = document.getElementById('micBtn');
                    if (!b || b.classList.contains('recording')) return;
                    window.toggleVoice(); // mic original do Chrome
                    liveMicActivated(true);
                    toast('Sua vez — pode falar.');
                } catch (e) {}
            }, 600);
        } catch (e) {}
    }

    async function onTurnFinished(expectVoice) {
        if (!state.enabled) return;
        const runId = state.runId;
        const text = lastBotText();
        if (!text || !text.trim()) {
            toast('Live: não encontrei texto na resposta para falar.', true);
            if (expectVoice) restartMicForNextTurn(runId);
            return;
        }
        state.stopFlag = false;
        const chunks = await splitText(text);
        if (!chunks.length || runId !== state.runId || !state.enabled) return;
        toast(`Falando resposta (${chunks.length} trecho${chunks.length > 1 ? 's' : ''})…`);
        await speakChunks(chunks.slice(0, 12), runId); // teto: 12 frases por resposta
        state.speaking = false;
        paintBtn();
        if (expectVoice) restartMicForNextTurn(runId);
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
            liveKeepOff();
            liveMicUntap();
            showLivePanel(false);
            stopAudio();
            try { // desliga o ditado se estava ouvindo
                const b = document.getElementById('micBtn');
                if (b && b.classList.contains('recording') && typeof window.toggleVoice === 'function') {
                    window.toggleVoice();
                }
            } catch (e) {}
            if (state.autoSendTimer) { clearTimeout(state.autoSendTimer); state.autoSendTimer = null; }
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
        showLivePanel(true);
        liveMicTap(); // visual da waveform (não grava, só mede o volume)
        paintBtn();
        toast('Live ligado — pode falar.');
        // já entra ouvindo: não precisa clicar no mic
        try {
            const b = document.getElementById('micBtn');
            if (b && !b.classList.contains('recording') && typeof window.toggleVoice === 'function') {
                setTimeout(() => {
                    try {
                        if (!state.enabled) return;
                        const b2 = document.getElementById('micBtn');
                        if (!b2 || b2.classList.contains('recording')) return;
                        window.toggleVoice();
                        liveMicActivated(true);
                    } catch (e) {}
                }, 400);
            }
        } catch (e) {}
    }

    // ── Mic no modo Live: grava -> transcreve (local) -> envia sozinho ──
    function micVisual(on) {
        try {
            const btn = document.getElementById('micBtn');
            if (btn) btn.classList.toggle('recording', !!on);
        } catch (e) {}
    }

    function stopRecorder() {
        return new Promise((resolve) => {
            const rec = state.recorder;
            if (!rec || rec.state === 'inactive') return resolve(null);
            rec.onstop = () => resolve(new Blob(state.recChunks, { type: rec.mimeType || 'audio/webm' }));
            try { rec.stop(); } catch (e) { resolve(null); }
            try {
                const st = state.recStream || rec.stream;
                if (st) st.getTracks().forEach(t => t.stop());
            } catch (e) {}
            state.recStream = null;
        });
    }

    async function uploadTranscribe(blob) {
        const fd = new FormData();
        fd.append('audio', blob, 'fala.webm');
        fd.append('language', 'pt');
        const res = await fetch(API.transcribe, { method: 'POST', body: fd });
        if (res.status === 503) {
            const data = await res.json().catch(() => ({}));
            toast(data.error || 'Transcrição indisponível. Rode: pip install faster-whisper', true);
            return '';
        }
        if (!res.ok) {
            const data = await res.json().catch(() => ({}));
            toast(data.error || `Falha ao transcrever (HTTP ${res.status}).`, true);
            return '';
        }
        const data = await res.json();
        return (data.text || '').trim();
    }

    async function listen() {
        // segundo clique (ou fim): para, transcreve e envia
        if (state.recording) {
            state.recording = false;
            clearTimeout(state.recTimer);
            micVisual(false);
            paintBtn();
            toast('Transcrevendo…');
            try {
                const blob = await stopRecorder();
                state.recorder = null;
                if (!blob || !blob.size) { toast('Nenhum áudio capturado.', true); return; }
                const text = await uploadTranscribe(blob);
                if (!text) return;
                const input = document.getElementById('promptInput');
                if (input) {
                    input.value = text;
                    input.dispatchEvent(new Event('input'));
                }
                toast(`Você disse: "${text.slice(0, 80)}${text.length > 80 ? '…' : ''}"`);
                if (typeof window.sendMessage === 'function') window.sendMessage();
            } catch (e) {
                console.warn('live listen erro:', e);
                toast('Falha ao processar o áudio.', true);
            }
            return;
        }
        // primeiro clique: começa a gravar
        try {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                toast('Navegador não permite microfone aqui. Use localhost ou HTTPS.', true);
                return;
            }
            if (typeof MediaRecorder === 'undefined') {
                toast('Navegador não suporta gravação de áudio.', true);
                return;
            }
            // se o navegador já negou antes, avisa como liberar em vez de falhar mudo
            try {
                if (navigator.permissions && navigator.permissions.query) {
                    const st = await navigator.permissions.query({ name: 'microphone' });
                    if (st.state === 'denied') {
                        toast('Mic bloqueado p/ este site. Clique no cadeado da barra de endereço → Microfone → Permitir, e recarregue (F5).', true);
                        return;
                    }
                }
            } catch (e) {}
            stopAudio();
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const mime = (window.MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported('audio/webm'))
                ? 'audio/webm' : '';
            const rec = mime ? new MediaRecorder(stream, { mimeType: mime }) : new MediaRecorder(stream);
            state.recChunks = [];
            rec.ondataavailable = (e) => { if (e.data && e.data.size) state.recChunks.push(e.data); };
            rec.start(250);
            state.recStream = stream; // MediaRecorder.stream é só-leitura: guarda separado
            state.recorder = rec;
            state.recording = true;
            micVisual(true);
            toast('Ouvindo… clique no mic de novo para enviar.');
            state.recTimer = setTimeout(() => { if (state.recording) listen(); }, REC_MAX_S * 1000);
        } catch (e) {
            console.warn('live mic erro:', e);
            micVisual(false);
            state.recording = false;
            const name = (e && e.name) || '';
            if (name === 'NotAllowedError') {
                toast('Permissão negada. Clique no cadeado da barra de endereço → Microfone → Permitir, e recarregue (F5).', true);
            } else if (name === 'NotFoundError') {
                toast('Nenhum microfone encontrado no computador.', true);
            } else if (name === 'NotReadableError') {
                toast('Microfone em uso por outro programa (Meet, Teams, etc.). Feche-o e tente de novo.', true);
            } else {
                toast('Microfone bloqueado. Permita o acesso no navegador.', true);
            }
        }
    }

    // Envolve o sendMessage original SEM alterá-lo: nova mensagem = interrompe fala atual.
    function wrapSendMessage() {
        try {
            if (typeof window.sendMessage !== 'function' || window.sendMessage.__liveWrapped) return;
            const orig = window.sendMessage;
            const wrapped = async function () {
                stopAudio(); // interrompe fala anterior ao enviar
                liveKeepOff(); // enviando: não mais aguardando fala
                state.stopFlag = false;
                state.busy = true; // trava o auto-envio enquanto gera/responde
                paintBtn(); // painel mostra "Pensando…"
                const myRun = state.runId;
                const myVoice = state.lastWasVoice === true; // consome a marca
                state.lastWasVoice = false;
                try {
                    return await orig.apply(this, arguments);
                } finally {
                    state.busy = false;
                    if (state.enabled && myRun === state.runId) onTurnFinished(myVoice);
                    else { state.speaking = false; paintBtn(); }
                }
            };
            wrapped.__liveWrapped = true;
            window.sendMessage = wrapped;
        } catch (e) { console.warn('live wrap falhou:', e); }
    }

    // O mic SEMPRE usa o ditado do navegador (Chrome): vai escrevendo na caixa
    // em tempo real. Sem wrapper aqui de propósito.
    // A transcrição local (LiveMode.listen -> /api/live/transcribe/) segue
    // disponível como alternativa, mas não é a rota padrão do botão.

    function liveKeepOff() { state.keepAlive.active = false; }
    function liveMicActivated(fresh) {
        state.keepAlive.active = true;
        if (fresh) { state.keepAlive.since = Date.now(); state.keepAlive.restarts = 0; }
    }

    // Mic parou SEM texto = timeout de silêncio do Chrome: religa sozinho.
    // Com texto = fim de fala: segue pro auto-envio normal.
    function liveKeepAlive() {
        const ka = state.keepAlive;
        if (!state.enabled || !ka.active || state.busy) return;
        const elapsed = (Date.now() - (ka.since || Date.now())) / 1000;
        if (ka.restarts >= KEEP_MAX_RESTARTS || elapsed > KEEP_MAX_S) {
            ka.active = false;
            toast('Mic em espera — clique para falar.');
            return;
        }
        ka.restarts += 1;
        if (ka.restarts === 1) toast('Continuo ouvindo…');
        setTimeout(() => {
            try {
                if (!state.enabled || !state.keepAlive.active || state.busy) return;
                const b = document.getElementById('micBtn');
                if (!b || b.classList.contains('recording')) return;
                if (typeof window.toggleVoice !== 'function') return;
                window.toggleVoice();
            } catch (e) {}
        }, 400);
    }

    // ── Auto-envio no Live: Chrome parou de ouvir -> envia sozinho ──
    // Observa a classe 'recording' do botão (que o script.js liga/desliga).
    // Só no Live, com ~1s de respiro (dá tempo de voltar a falar e cancelar).
    function setupMicAutoSend() {
        try {
            const btn = document.getElementById('micBtn');
            const input = document.getElementById('promptInput');
            if (!btn || !input || btn.__liveObserved) return;
            btn.__liveObserved = true;
            // clique do usuário: ligou = quer conversar (keep-alive); desligou = respeita
            btn.addEventListener('click', () => setTimeout(() => {
                try {
                    if (!state.enabled) { liveKeepOff(); return; }
                    if (btn.classList.contains('recording')) liveMicActivated(true);
                    else {
                        liveKeepOff();
                        if (state.autoSendTimer) {
                            clearTimeout(state.autoSendTimer);
                            state.autoSendTimer = null;
                        }
                    }
                } catch (e) {}
            }, 0));
            let wasRecording = btn.classList.contains('recording');
            const obs = new MutationObserver(() => {
                const isRec = btn.classList.contains('recording');
                paintBtn(); // atualiza status/wave do painel
                if (isRec) {
                    // voltou a falar: cancela envio pendente
                    wasRecording = true;
                    if (state.autoSendTimer) {
                        clearTimeout(state.autoSendTimer);
                        state.autoSendTimer = null;
                    }
                    return;
                }
                if (wasRecording && !isRec) {
                    // parou de ouvir
                    wasRecording = false;
                    if (!state.enabled) { liveKeepOff(); return; } // normal: manual
                    const textNow = ((document.getElementById('promptInput') || {}).value || '').trim();
                    if (textNow.length >= 2) {
                        liveKeepOff(); // vai enviar; o loop reativa depois da resposta
                        if (state.autoSendTimer) clearTimeout(state.autoSendTimer);
                        state.autoSendTimer = setTimeout(() => {
                            state.autoSendTimer = null;
                            if (!state.enabled || state.busy) return;
                            const text = (document.getElementById('promptInput') || {}).value || '';
                            if (text.trim().length < 2) return;
                            state.lastWasVoice = true; // marca: turno veio do mic
                            if (typeof window.sendMessage === 'function') window.sendMessage();
                        }, 1000);
                    } else {
                        liveKeepAlive(); // silêncio: mantém o mic vivo
                    }
                }
            });
            obs.observe(btn, { attributes: true, attributeFilter: ['class'] });
        } catch (e) { console.warn('live autosend falhou:', e); }
    }

    // script.js define window.sendMessage no fim do arquivo; retry.
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
        listen,
        get enabled() { return state.enabled; },
        setVoice(v) { state.voice = v; },
        setSpeed(s) { state.speed = Math.max(0.5, Math.min(2.0, Number(s) || 1.0)); },
    };

    document.addEventListener('DOMContentLoaded', () => { paintBtn(); setupMicAutoSend(); });
    try { setupMicAutoSend(); } catch (e) {}
})();
