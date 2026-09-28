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
        speed: 0.95, // fala mais ágil e natural (0.95x)
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
        if (!isErr) return; // Não exibir notificações informativas perto do microfone
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
            const sub = document.getElementById('liveStatusSub');
            const orbWrapper = document.getElementById('geminiOrbWrapper');

            if (state.muted) {
                if (st) st.textContent = 'Microfone em pausa';
                if (sub) sub.textContent = 'Clique no ícone de microfone para reativar';
            } else if (state.speaking) {
                if (st) st.textContent = 'Falando…';
                if (sub) sub.textContent = 'Ouvindo a resposta sintetizada';
            } else if (listening && !state.speaking) {
                if (st) st.textContent = 'Ouvindo você…';
                if (sub) sub.textContent = 'Fale normalmente, o microfone está ativo';
            } else if (state.busy) {
                if (st) st.textContent = 'Pensando…';
                if (sub) sub.textContent = 'Analisando o contexto e gerando resposta';
            } else {
                if (st) st.textContent = 'Pode falar';
                if (sub) sub.textContent = 'Aguardando sua pergunta ou comando de voz';
            }

            if (orbWrapper) {
                orbWrapper.classList.toggle('speaking', state.speaking);
                orbWrapper.classList.toggle('listening', listening && !state.speaking && !state.muted);
                orbWrapper.classList.toggle('thinking', state.busy && !state.speaking);
                orbWrapper.classList.toggle('muted', !!state.muted);
            }

            const engineName = document.getElementById('geminiEngineName');
            if (engineName && state.voice) {
                const isEdge = state.voice.startsWith('pt-BR-');
                engineName.textContent = isEdge ? 'Microsoft Neural' : 'Piper Local';
            }
        } catch (e) {}
    }

    // ── Painel Live: substitui a tela principal (mantendo a barra lateral) ──
    function buildLivePanel() {
        try {
            if (document.getElementById('livePanel')) return;
            const main = document.querySelector('.main');
            if (!main) return;
            const panel = document.createElement('div');
            panel.id = 'livePanel';
            panel.className = 'gemini-live-panel';
            panel.style.display = 'none';
            panel.innerHTML =
                '<div class="gemini-live-header">' +
                    '<div class="gemini-live-badge">' +
                        '<span class="gemini-live-pulse-dot"></span>' +
                        '<span>Modo Live Ativo</span>' +
                    '</div>' +
                '</div>' +

                '<div class="gemini-center-stage">' +
                    '<div class="gemini-orb-wrapper" id="geminiOrbWrapper">' +
                        '<div class="gemini-orb-ambient-glow" id="geminiOrbGlow"></div>' +
                        '<div class="gemini-orb-rings">' +
                            '<div class="gemini-ring ring-1"></div>' +
                            '<div class="gemini-ring ring-2"></div>' +
                            '<div class="gemini-ring ring-3"></div>' +
                        '</div>' +
                        '<canvas id="liveOrbCanvas" width="280" height="280"></canvas>' +
                    '</div>' +

                    '<div class="gemini-status-block">' +
                        '<div class="gemini-status-title" id="liveStatus">Ouvindo você…</div>' +
                        '<div class="gemini-status-sub" id="liveStatusSub">Fale normalmente, o microfone está ativo</div>' +
                    '</div>' +
                '</div>' +

                '<div class="gemini-dock">' +
                    '<button class="gemini-end-btn" id="liveEndBtn" title="Encerrar Modo Live">' +
                        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" width="18" height="18">' +
                            '<line x1="18" y1="6" x2="6" y2="18"/>' +
                            '<line x1="6" y1="6" x2="18" y2="18"/>' +
                        '</svg>' +
                        '<span>Encerrar</span>' +
                    '</button>' +
                '</div>';

            main.appendChild(panel);

            const endBtn = document.getElementById('liveEndBtn');
            if (endBtn) endBtn.addEventListener('click', () => toggle());

            startOrbAnimationLoop();
        } catch (e) { console.warn('live panel falhou:', e); }
    }

    function showLivePanel(on) {
        try {
            buildLivePanel();
            const panel = document.getElementById('livePanel');
            const zone = document.querySelector('.input-zone');
            const hero = document.getElementById('welcomeHero');
            const chat = document.getElementById('chatArea');
            const topbar = document.querySelector('.topbar');

            if (on) {
                if (zone) zone.style.display = 'none';
                if (hero) hero.style.display = 'none';
                if (chat) chat.style.display = 'none';
                if (topbar) topbar.style.display = 'none';
                if (panel) panel.style.display = 'flex';
            } else {
                if (panel) panel.style.display = 'none';
                if (topbar) topbar.style.display = '';
                if (zone) zone.style.display = '';
                if (typeof renderCurrentMessages === 'function') {
                    renderCurrentMessages();
                } else {
                    if (chat && chat.children.length > 0) {
                        chat.style.display = 'block';
                        if (hero) hero.style.display = 'none';
                    } else if (hero) {
                        hero.style.display = 'flex';
                    }
                }
            }
        } catch (e) {
            console.warn('showLivePanel erro:', e);
        }
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

    // ── Efeitos Sonoros Sutis de Início e Fim (Web Audio Sintetizado) ──
    function playLiveChime(type) {
        try {
            const ctx = liveEnsureCtx();
            if (!ctx) return;
            const now = ctx.currentTime;

            if (type === 'start') {
                // Chime de Ativação: Acorde arpeggiado ascendente futurista (C5 -> E5 -> G5 -> C6)
                const notes = [
                    { f: 523.25, t: 0.00, d: 0.14, v: 0.16 }, // C5
                    { f: 659.25, t: 0.07, d: 0.16, v: 0.18 }, // E5
                    { f: 783.99, t: 0.14, d: 0.18, v: 0.20 }, // G5
                    { f: 1046.50, t: 0.21, d: 0.32, v: 0.22 }, // C6
                ];
                notes.forEach(n => {
                    const osc = ctx.createOscillator();
                    const gain = ctx.createGain();
                    osc.type = 'sine';
                    osc.frequency.setValueAtTime(n.f, now + n.t);

                    gain.gain.setValueAtTime(0.0001, now + n.t);
                    gain.gain.exponentialRampToValueAtTime(n.v, now + n.t + 0.015);
                    gain.gain.exponentialRampToValueAtTime(0.0001, now + n.t + n.d);

                    osc.connect(gain);
                    gain.connect(ctx.destination);

                    osc.start(now + n.t);
                    osc.stop(now + n.t + n.d + 0.05);
                });
            } else if (type === 'end') {
                // Chime de Desativação: Dois tons descendentes suaves e aveludados (G5 -> C5)
                const notes = [
                    { f: 783.99, t: 0.00, d: 0.14, v: 0.16 }, // G5
                    { f: 523.25, t: 0.09, d: 0.28, v: 0.14 }, // C5
                ];
                notes.forEach(n => {
                    const osc = ctx.createOscillator();
                    const gain = ctx.createGain();
                    osc.type = 'sine';
                    osc.frequency.setValueAtTime(n.f, now + n.t);

                    gain.gain.setValueAtTime(0.0001, now + n.t);
                    gain.gain.exponentialRampToValueAtTime(n.v, now + n.t + 0.015);
                    gain.gain.exponentialRampToValueAtTime(0.0001, now + n.t + n.d);

                    osc.connect(gain);
                    gain.connect(ctx.destination);

                    osc.start(now + n.t);
                    osc.stop(now + n.t + n.d + 0.05);
                });
            }
        } catch (e) {
            console.warn('Erro ao tocar efeito sonoro live:', e);
        }
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

    let orbT = 0;
    const orbParticles = Array.from({ length: 18 }, (_, i) => ({
        angle: (i / 18) * Math.PI * 2,
        dist: 78 + Math.random() * 36,
        speed: 0.007 + Math.random() * 0.012,
        size: 1.5 + Math.random() * 2,
        alpha: 0.35 + Math.random() * 0.5,
    }));

    function startOrbAnimationLoop() {
        try {
            const cv = document.getElementById('liveOrbCanvas');
            if (!cv) return;
            const cx = cv.getContext('2d');

            function frame() {
                try {
                    const panel = document.getElementById('livePanel');
                    if (panel && panel.style.display !== 'none') {
                        const W = cv.width, H = cv.height;
                        const centerX = W / 2, centerY = H / 2;
                        cx.clearRect(0, 0, W, H);

                        const { bars, live } = waveLevels(32);
                        let avgLevel = 0;
                        if (live && bars.length) {
                            avgLevel = bars.reduce((a, b) => a + b, 0) / bars.length;
                        }
                        if (state.muted) avgLevel = 0;

                        orbT += 0.028;

                        const isLight = document.documentElement.getAttribute('data-theme') === 'light';

                        // Cores do orbe adaptadas por estado e tema
                        let pRGB, sRGB, aRGB;
                        if (state.muted) {
                            pRGB = isLight ? [100, 116, 139] : [148, 163, 184];
                            sRGB = isLight ? [148, 163, 184] : [100, 116, 139];
                            aRGB = isLight ? [71, 85, 105] : [51, 65, 85];
                        } else if (state.speaking) {
                            pRGB = isLight ? [2, 132, 199] : [6, 182, 212];
                            sRGB = isLight ? [37, 99, 235] : [59, 130, 246];
                            aRGB = isLight ? [13, 148, 136] : [34, 197, 94];
                        } else if (state.busy) {
                            pRGB = isLight ? [147, 51, 234] : [168, 85, 247];
                            sRGB = isLight ? [219, 39, 119] : [236, 72, 153];
                            aRGB = isLight ? [37, 99, 235] : [59, 130, 246];
                        } else {
                            pRGB = isLight ? [16, 185, 129] : [34, 197, 94];
                            sRGB = isLight ? [5, 150, 105] : [16, 185, 129];
                            aRGB = isLight ? [2, 132, 199] : [6, 182, 212];
                        }

                        // 1. Partículas orbitais sutis (poeira estelar)
                        for (let p of orbParticles) {
                            p.angle += p.speed;
                            const px = centerX + Math.cos(p.angle) * p.dist;
                            const py = centerY + Math.sin(p.angle) * p.dist;
                            cx.beginPath();
                            cx.arc(px, py, p.size, 0, Math.PI * 2);
                            cx.fillStyle = `rgba(${pRGB[0]}, ${pRGB[1]}, ${pRGB[2]}, ${p.alpha * (isLight ? 0.45 : 0.65)})`;
                            cx.fill();
                        }

                        // 2. Halo / Aura difusa externa
                        const auraR = 86 + (avgLevel * 18) + Math.sin(orbT * 1.2) * 3;
                        const auraGrad = cx.createRadialGradient(centerX, centerY, 48, centerX, centerY, auraR);
                        const auraAlpha = isLight ? 0.22 : 0.35;
                        auraGrad.addColorStop(0, `rgba(${pRGB[0]}, ${pRGB[1]}, ${pRGB[2]}, ${auraAlpha})`);
                        auraGrad.addColorStop(0.55, `rgba(${sRGB[0]}, ${sRGB[1]}, ${sRGB[2]}, ${auraAlpha * 0.4})`);
                        auraGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
                        cx.fillStyle = auraGrad;
                        cx.beginPath();
                        cx.arc(centerX, centerY, auraR, 0, Math.PI * 2);
                        cx.fill();

                        // 3. Raio Perfeito e Circular (sem contornos de ameba / desenho animado)
                        const pulse = (live || state.speaking) ? (avgLevel * 9) : Math.sin(orbT * 1.4) * 1.6;
                        const R = 68 + pulse;

                        // 4. Esfera 3D Volumétrica com Dinâmica Interna de Plasma
                        cx.save();
                        // CLIP RIGOROSAMENTE CIRCULAR
                        cx.beginPath();
                        cx.arc(centerX, centerY, R, 0, Math.PI * 2);
                        cx.clip();

                        // a) Base 3D esférica com profundidade óptica
                        const sphereGrad = cx.createRadialGradient(
                            centerX - R * 0.28, centerY - R * 0.32, R * 0.05,
                            centerX, centerY, R
                        );
                        if (isLight) {
                            sphereGrad.addColorStop(0, '#ffffff');
                            sphereGrad.addColorStop(0.22, `rgba(${pRGB[0]}, ${pRGB[1]}, ${pRGB[2]}, 0.88)`);
                            sphereGrad.addColorStop(0.68, `rgba(${sRGB[0]}, ${sRGB[1]}, ${sRGB[2]}, 0.95)`);
                            sphereGrad.addColorStop(1, `rgba(${aRGB[0]}, ${aRGB[1]}, ${aRGB[2]}, 1)`);
                        } else {
                            sphereGrad.addColorStop(0, '#ffffff');
                            sphereGrad.addColorStop(0.2, `rgba(${pRGB[0]}, ${pRGB[1]}, ${pRGB[2]}, 0.92)`);
                            sphereGrad.addColorStop(0.65, `rgba(${sRGB[0]}, ${sRGB[1]}, ${sRGB[2]}, 0.85)`);
                            sphereGrad.addColorStop(1, `rgba(${aRGB[0]}, ${aRGB[1]}, ${aRGB[2]}, 0.55)`);
                        }
                        cx.fillStyle = sphereGrad;
                        cx.fillRect(centerX - R, centerY - R, R * 2, R * 2);

                        // b) Vórtices internos em movimento caustico
                        cx.save();
                        cx.globalCompositeOperation = 'screen';

                        const speedMult = (state.speaking || live) ? 1.7 : 1.0;
                        const vx1 = centerX + Math.cos(orbT * 1.1 * speedMult) * (R * 0.26);
                        const vy1 = centerY + Math.sin(orbT * 0.85 * speedMult) * (R * 0.22);
                        const vg1 = cx.createRadialGradient(vx1, vy1, 2, vx1, vy1, R * 0.65);
                        vg1.addColorStop(0, `rgba(255, 255, 255, ${0.5 + avgLevel * 0.4})`);
                        vg1.addColorStop(0.5, `rgba(${pRGB[0]}, ${pRGB[1]}, ${pRGB[2]}, 0.4)`);
                        vg1.addColorStop(1, 'rgba(0, 0, 0, 0)');
                        cx.fillStyle = vg1;
                        cx.beginPath();
                        cx.arc(vx1, vy1, R * 0.65, 0, Math.PI * 2);
                        cx.fill();

                        const vx2 = centerX + Math.cos(-orbT * 1.3 * speedMult + 2.2) * (R * 0.3);
                        const vy2 = centerY + Math.sin(-orbT * 1.05 * speedMult + 1.4) * (R * 0.26);
                        const vg2 = cx.createRadialGradient(vx2, vy2, 2, vx2, vy2, R * 0.6);
                        vg2.addColorStop(0, `rgba(${aRGB[0]}, ${aRGB[1]}, ${aRGB[2]}, ${0.45 + avgLevel * 0.3})`);
                        vg2.addColorStop(0.6, `rgba(${sRGB[0]}, ${sRGB[1]}, ${sRGB[2]}, 0.25)`);
                        vg2.addColorStop(1, 'rgba(0, 0, 0, 0)');
                        cx.fillStyle = vg2;
                        cx.beginPath();
                        cx.arc(vx2, vy2, R * 0.6, 0, Math.PI * 2);
                        cx.fill();

                        // c) Núcleo reativo à frequência da voz
                        const coreR = (R * 0.22) + (avgLevel * R * 0.24);
                        const coreGrad = cx.createRadialGradient(centerX, centerY, 0, centerX, centerY, coreR);
                        coreGrad.addColorStop(0, 'rgba(255, 255, 255, 0.92)');
                        coreGrad.addColorStop(0.4, `rgba(${pRGB[0]}, ${pRGB[1]}, ${pRGB[2]}, 0.65)`);
                        coreGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
                        cx.fillStyle = coreGrad;
                        cx.beginPath();
                        cx.arc(centerX, centerY, coreR, 0, Math.PI * 2);
                        cx.fill();

                        cx.restore();

                        // d) Fresnel Rim Light (Luz de borda reflexiva interna)
                        const rimGrad = cx.createRadialGradient(centerX, centerY, R * 0.72, centerX, centerY, R);
                        rimGrad.addColorStop(0, 'rgba(255, 255, 255, 0)');
                        rimGrad.addColorStop(0.85, 'rgba(255, 255, 255, 0.15)');
                        rimGrad.addColorStop(1, 'rgba(255, 255, 255, 0.55)');
                        cx.fillStyle = rimGrad;
                        cx.fillRect(centerX - R, centerY - R, R * 2, R * 2);

                        // e) Reflexo Especular Superior de Vidro
                        cx.save();
                        cx.beginPath();
                        cx.ellipse(centerX - R * 0.22, centerY - R * 0.32, R * 0.38, R * 0.16, -Math.PI / 5, 0, Math.PI * 2);
                        const specGrad = cx.createLinearGradient(
                            centerX - R * 0.45, centerY - R * 0.45,
                            centerX - R * 0.05, centerY - R * 0.15
                        );
                        specGrad.addColorStop(0, 'rgba(255, 255, 255, 0.75)');
                        specGrad.addColorStop(0.6, 'rgba(255, 255, 255, 0.18)');
                        specGrad.addColorStop(1, 'rgba(255, 255, 255, 0)');
                        cx.fillStyle = specGrad;
                        cx.fill();
                        cx.restore();

                        cx.restore(); // fim do clip circular

                        // 5. Linha de Borda Sutil e Fina
                        cx.save();
                        cx.beginPath();
                        cx.arc(centerX, centerY, R, 0, Math.PI * 2);
                        cx.lineWidth = 1.0;
                        cx.strokeStyle = isLight ? 'rgba(0, 0, 0, 0.09)' : 'rgba(255, 255, 255, 0.28)';
                        cx.stroke();
                        cx.restore();
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
            playLiveChime('end');
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
            return;
        }
        // ligando: verifica motor primeiro (primeira voz pode baixar ~60MB)
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
                const savedVoice = localStorage.getItem('live_voice_pref');
                const voiceSel = document.getElementById('liveVoiceSelect');
                if (voiceSel && Array.isArray(data.voices)) {
                    voiceSel.innerHTML = '';
                    data.voices.forEach(v => {
                        const opt = document.createElement('option');
                        opt.value = v.id;
                        opt.textContent = v.name || v.id;
                        voiceSel.appendChild(opt);
                    });
                }
                if (savedVoice && data.voices && data.voices.some(v => v.id === savedVoice)) {
                    state.voice = savedVoice;
                } else if (data.default) {
                    state.voice = data.default;
                }
                if (voiceSel && state.voice) {
                    voiceSel.value = state.voice;
                }
            }
        } catch (e) { console.warn('Falha ao listar vozes:', e); }
        showLivePanel(true);
        playLiveChime('start');
        liveMicTap(); // visual da waveform (não grava, só mede o volume)
        paintBtn();
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

    // ── Gestão do Modo Live no Modal de Configurações ──
    async function initLiveSettingsTab() {
        try {
            const voiceSelect = document.getElementById('modalLiveVoiceSelect');
            const speedRange = document.getElementById('liveSpeedRange');
            const speedDisplay = document.getElementById('liveSpeedDisplay');

            const savedSpeed = parseFloat(localStorage.getItem('live_speed_pref')) || state.speed || 0.95;
            state.speed = savedSpeed;
            if (speedRange) speedRange.value = savedSpeed;
            if (speedDisplay) speedDisplay.textContent = savedSpeed.toFixed(2) + 'x';

            const res = await fetch(API.voices);
            if (res.ok) {
                const data = await res.json();
                if (voiceSelect && Array.isArray(data.voices)) {
                    voiceSelect.innerHTML = '';
                    const savedVoice = localStorage.getItem('live_voice_pref') || state.voice || data.default;
                    data.voices.forEach(v => {
                        const opt = document.createElement('option');
                        opt.value = v.id;
                        const icon = v.engine === 'edge-tts' ? '🌐' : '💻';
                        opt.textContent = `${icon} ${v.name || v.id}`;
                        voiceSelect.appendChild(opt);
                    });
                    if (savedVoice) {
                        voiceSelect.value = savedVoice;
                        state.voice = savedVoice;
                    }
                    onModalLiveVoiceChange(voiceSelect.value);
                }
            }

            const hRes = await fetch(API.health);
            if (hRes.ok) {
                const hData = await hRes.json();
                const edgeBadge = document.getElementById('liveEdgeStatusBadge');
                const piperBadge = document.getElementById('livePiperStatusBadge');
                const sttBadge = document.getElementById('liveSttStatusBadge');
                if (edgeBadge && hData.live) {
                    edgeBadge.textContent = hData.live.edge_tts ? 'Ativo' : 'Indisponível';
                    edgeBadge.style.color = hData.live.edge_tts ? '#22c55e' : '#f59e0b';
                }
                if (piperBadge && hData.live) {
                    piperBadge.textContent = (hData.live.python_lib || hData.live.cli) ? 'Pronto (Fallback)' : 'Não instalado';
                    piperBadge.style.color = (hData.live.python_lib || hData.live.cli) ? '#3b82f6' : 'var(--text-dim)';
                }
                if (sttBadge && hData.stt) {
                    sttBadge.textContent = hData.stt.available ? 'Pronto' : 'Indisponível';
                    sttBadge.style.color = hData.stt.available ? '#22c55e' : 'var(--text-dim)';
                }
            }
        } catch (e) {
            console.warn('initLiveSettingsTab falhou:', e);
        }
    }

    function onModalLiveVoiceChange(val) {
        const desc = document.getElementById('modalLiveVoiceDesc');
        if (!desc) return;
        if (val && val.startsWith('pt-BR-')) {
            desc.innerHTML = '<span style="color: #22c55e; font-weight: 600;">● Voz Neural (Nuvem):</span> Alta naturalidade, entonação e expressividade humana.';
        } else {
            desc.innerHTML = '<span style="color: #3b82f6; font-weight: 600;">● Voz Piper (Local):</span> 100% offline, processada diretamente no computador.';
        }
    }

    function updateLiveSpeedPreview(val) {
        const disp = document.getElementById('liveSpeedDisplay');
        const num = parseFloat(val) || 0.95;
        if (disp) disp.textContent = num.toFixed(2) + 'x';
    }

    let sampleAudio = null;
    async function testLiveVoiceSample() {
        const btn = document.getElementById('testVoiceBtn');
        const btnText = document.getElementById('testVoiceBtnText');
        const voiceSelect = document.getElementById('modalLiveVoiceSelect');
        const speedRange = document.getElementById('liveSpeedRange');

        const voice = voiceSelect ? voiceSelect.value : state.voice;
        const speed = speedRange ? parseFloat(speedRange.value) : state.speed;

        if (sampleAudio) {
            try { sampleAudio.pause(); } catch (e) {}
            sampleAudio = null;
        }

        if (btnText) btnText.textContent = 'Gerando amostra…';
        if (btn) btn.disabled = true;

        try {
            const res = await fetch(API.speak, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: 'Olá! Esta é uma demonstração da síntese de voz no Modo Live.',
                    voice: voice,
                    speed: speed,
                }),
            });
            if (!res.ok) {
                toast('Erro ao gerar amostra: HTTP ' + res.status, true);
                if (btnText) btnText.textContent = 'Ouvir Amostra da Voz';
                if (btn) btn.disabled = false;
                return;
            }
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            sampleAudio = new Audio(url);
            sampleAudio.play();
            const resetBtn = () => {
                if (btnText) btnText.textContent = 'Ouvir Amostra da Voz';
                if (btn) btn.disabled = false;
                setTimeout(() => URL.revokeObjectURL(url), 5000);
            };
            sampleAudio.onended = resetBtn;
            sampleAudio.onerror = resetBtn;
        } catch (e) {
            toast('Falha ao reproduzir amostra: ' + e, true);
            if (btnText) btnText.textContent = 'Ouvir Amostra da Voz';
            if (btn) btn.disabled = false;
        }
    }

    function saveLiveSettingsFromModal() {
        const voiceSelect = document.getElementById('modalLiveVoiceSelect');
        const speedRange = document.getElementById('liveSpeedRange');
        const feedback = document.getElementById('liveSettingsFeedback');

        const chosenVoice = voiceSelect ? voiceSelect.value : state.voice;
        const chosenSpeed = speedRange ? parseFloat(speedRange.value) : state.speed;

        if (chosenVoice) {
            state.voice = chosenVoice;
            try { localStorage.setItem('live_voice_pref', chosenVoice); } catch (e) {}
            const liveVoiceSelect = document.getElementById('liveVoiceSelect');
            if (liveVoiceSelect) liveVoiceSelect.value = chosenVoice;
        }

        if (chosenSpeed) {
            state.speed = chosenSpeed;
            try { localStorage.setItem('live_speed_pref', chosenSpeed); } catch (e) {}
        }

        if (feedback) {
            feedback.style.display = 'block';
            feedback.style.color = '#22c55e';
            feedback.textContent = '✓ Preferências de voz salvas com sucesso!';
            setTimeout(() => { feedback.style.display = 'none'; }, 3000);
        }
        toast('Configurações do Modo Live salvas.');
    }

    window.initLiveSettingsTab = initLiveSettingsTab;
    window.onModalLiveVoiceChange = onModalLiveVoiceChange;
    window.updateLiveSpeedPreview = updateLiveSpeedPreview;
    window.testLiveVoiceSample = testLiveVoiceSample;
    window.saveLiveSettingsFromModal = saveLiveSettingsFromModal;

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
