"""Serviço TTS (Piper) do modo Live — singleton isolado do LMMService.

Não importa nada de api/services. Falhas aqui nunca afetam o chat.
"""
import asyncio
import concurrent.futures
import hashlib
import logging
import re
import shutil
import threading
import urllib.request
import wave
from pathlib import Path

from django.conf import settings

logger = logging.getLogger(__name__)

BASE_DIR = Path(settings.BASE_DIR)

# Catálogo de vozes neurais de alta fidelidade (Edge-TTS - Microsoft Neural)
EDGE_VOICES = {
    'pt-BR-AntonioNeural': {
        'id': 'pt-BR-AntonioNeural',
        'name': 'Antonio (Masculina Neural - Muito Natural)',
        'gender': 'male',
        'engine': 'edge-tts',
    },
    'pt-BR-FranciscaNeural': {
        'id': 'pt-BR-FranciscaNeural',
        'name': 'Francisca (Feminina Neural - Muito Natural)',
        'gender': 'female',
        'engine': 'edge-tts',
    },
    'pt-BR-ThalitaMultilingualNeural': {
        'id': 'pt-BR-ThalitaMultilingualNeural',
        'name': 'Thalita (Feminina Neural - Jovem)',
        'gender': 'female',
        'engine': 'edge-tts',
    },
}

# Alias amigáveis de voz (ex: 'fabio' -> 'pt_BR-faber-medium')
VOICE_ALIASES = {
    'fabio': 'pt_BR-faber-medium',
    'faber': 'pt_BR-faber-medium',
    'antonio': 'pt-BR-AntonioNeural',
    'francisca': 'pt-BR-FranciscaNeural',
    'thalita': 'pt-BR-ThalitaMultilingualNeural',
    'cadu': 'pt_BR-cadu-medium',
}

# Voz padrão (pode trocar via .env: LIVE_VOICE=pt_BR-faber-medium ou pt-BR-AntonioNeural)
DEFAULT_VOICE = getattr(settings, 'LIVE_VOICE', getattr(settings, 'PIPER_VOICE', 'pt_BR-faber-medium'))
VOICE_DIR = Path(getattr(settings, 'PIPER_MODEL_DIR', BASE_DIR / 'models' / 'tts'))
CACHE_DIR = Path(getattr(settings, 'PIPER_CACHE_DIR', BASE_DIR / 'storage' / 'tts_cache'))
MAX_CHARS = int(getattr(settings, 'LIVE_MAX_CHARS', 600))

# Afinação de naturalidade (Piper). Ajuste via .env sem tocar no código.
# noise_scale maior = fala mais expressiva (padrão do Piper: 0.667).
DEFAULT_NOISE_SCALE = float(getattr(settings, 'PIPER_NOISE_SCALE', 0.8))
DEFAULT_NOISE_W = float(getattr(settings, 'PIPER_NOISE_W', 0.8))

HF_BASE = 'https://huggingface.co/rhasspy/piper-voices/resolve/main'

# Catálogo mínimo pt-BR local (Piper ONNX + JSON)
VOICE_CATALOG = {
    'pt_BR-faber-medium': (
        f'{HF_BASE}/pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx',
        f'{HF_BASE}/pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx.json',
    ),
    'pt_BR-cadu-medium': (
        f'{HF_BASE}/pt/pt_BR/cadu/medium/pt_BR-cadu-medium.onnx',
        f'{HF_BASE}/pt/pt_BR/cadu/medium/pt_BR-cadu-medium.onnx.json',
    ),
}


class PiperNotAvailable(Exception):
    pass


# ── Números por extenso (pt-BR) ──
# O Piper lê "18" como "um oito" às vezes; por extenso a prosódia acerta.
_PT_UNITS = (
    'zero', 'um', 'dois', 'três', 'quatro', 'cinco', 'seis', 'sete',
    'oito', 'nove', 'dez', 'onze', 'doze', 'treze', 'quatorze', 'quinze',
    'dezesseis', 'dezessete', 'dezoito', 'dezenove',
)
_PT_TENS = ('', '', 'vinte', 'trinta', 'quarenta', 'cinquenta', 'sessenta',
            'setenta', 'oitenta', 'noventa')
_PT_HUND = ('', 'cento', 'duzentos', 'trezentos', 'quatrocentos', 'quinhentos',
            'seiscentos', 'setecentos', 'oitocentos', 'novecentos')


def _pt_under_100(n: int) -> str:
    if n < 20:
        return _PT_UNITS[n]
    t, r = divmod(n, 10)
    return _PT_TENS[t] if r == 0 else f'{_PT_TENS[t]} e {_PT_UNITS[r]}'


def _pt_under_1000(n: int) -> str:
    if n == 100:
        return 'cem'
    h, r = divmod(n, 100)
    head = _PT_HUND[h] if h else ''
    if not head:
        return _pt_under_100(r)
    return head if r == 0 else f'{head} e {_pt_under_100(r)}'


def int_to_pt(n: int) -> str:
    """Inteiro até 999999 por extenso. Fora disso, devolve os dígitos."""
    if n < 0:
        return 'menos ' + int_to_pt(-n)
    if n < 1000:
        return _pt_under_1000(n)
    if n > 999999:
        return str(n)
    th, r = divmod(n, 1000)
    head = 'mil' if th == 1 else f'{_pt_under_1000(th)} mil'
    if r == 0:
        return head
    tail = _pt_under_1000(r)
    # "mil e duzentos" / "mil e cinco", mas "mil duzentos e cinquenta"
    return f'{head} e {tail}' if (r < 100 or r % 100 == 0) else f'{head} {tail}'


def _num_to_pt(raw: str) -> str:
    """'18,5' -> 'dezoito vírgula cinco' (dígito a dígito p/ não perder precisão)."""
    raw = raw.strip()
    if ',' in raw or '.' in raw:
        sep = ',' if ',' in raw else '.'
        ip, dp = raw.split(sep, 1)
        head = int_to_pt(int(ip)) if ip else 'zero'
        digits = ' '.join(_PT_UNITS[int(d)] for d in dp if d.isdigit())
        return f'{head} vírgula {digits}' if digits else head
    return int_to_pt(int(raw))


def expand_numbers_pt(text: str) -> str:
    """Expande % °C decimais e inteiros p/ palavras. Roda após a limpeza."""
    t = text
    # milhares pt-BR: 1.200 -> 1200
    t = re.sub(r'\b\d{1,3}(?:\.\d{3})+\b', lambda m: m.group(0).replace('.', ''), t)
    # temperatura: 25°C -> vinte e cinco graus
    t = re.sub(r'(\d+(?:[,.]\d+)?)\s*°C', lambda m: _num_to_pt(m.group(1)) + ' graus', t)
    t = re.sub(r'(\d+(?:[,.]\d+)?)\s*°(?!\w)', lambda m: _num_to_pt(m.group(1)) + ' graus', t)
    # percentual: 18% -> dezoito por cento
    t = re.sub(r'(\d+(?:[,.]\d+)?)\s*%', lambda m: _num_to_pt(m.group(1)) + ' por cento', t)
    # decimais restantes: 18,5
    t = re.sub(r'\b\d+[,.]\d+\b', lambda m: _num_to_pt(m.group(0)), t)
    # inteiros: 42 (até 999999; maiores ficam como dígitos)
    t = re.sub(r'\b\d+\b', lambda m: int_to_pt(int(m.group(0))), t)
    return t


def clean_text_for_speech(text: str) -> str:
    """Remove markdown/código/URLs que o TTS leria em voz alta."""
    if not text:
        return ''
    t = text
    # blocos de código ```...``` -> remove (ou troca por "código omitido" se for tudo código)
    t = re.sub(r'```.*?```', ' ', t, flags=re.DOTALL)
    # código inline `x` -> x
    t = re.sub(r'`([^`]*)`', r'\1', t)
    # imagens ![alt](url) -> alt
    t = re.sub(r'!\[([^\]]*)\]\([^)]*\)', r'\1', t)
    # links [texto](url) -> texto
    t = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', t)
    # headers ### -> texto
    t = re.sub(r'(?m)^\s{0,3}#{1,6}\s*', '', t)
    # negrito/itálico **x**, *x*, __x__, _x_ -> x
    t = re.sub(r'(\*\*|__)(.*?)\1', r'\2', t)
    t = re.sub(r'(?<!\w)(\*|_)(.*?)\1(?!\w)', r'\2', t)
    # html tags
    t = re.sub(r'<[^>]+>', ' ', t)
    # urls cruas
    t = re.sub(r'https?://\S+', ' ', t)
    # tabelas markdown: | a | b | -> a, b
    t = re.sub(r'\|', ', ', t)
    # bullets no início da linha
    t = re.sub(r'(?m)^\s*[-*•]\s+', '', t)
    # emojis básicos (fora do BMP comum) -> espaço
    t = re.sub(r'[\U00010000-\U0010ffff]', ' ', t)
    # números por extenso: a prosódia do Piper segue palavras, não dígitos
    t = expand_numbers_pt(t)
    # colapsa espaços
    t = re.sub(r'[ \t]+', ' ', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    t = t.strip()
    # ponto final garantido: sem pontuação terminal o Piper "engole" o fim da frase
    if t and t[-1] not in '.!?…:;':
        t += '.'
    return t


def split_into_chunks(text: str, limit: int = 220) -> list:
    """Divide texto limpo em frases curtas p/ síntese sequencial no frontend."""
    clean = clean_text_for_speech(text)
    if not clean:
        return []
    # quebra por fim de frase, depois por tamanho
    parts = re.split(r'(?<=[.!?…])\s+|\n+', clean)
    chunks = []
    buf = ''
    for p in parts:
        p = p.strip(' ,;:\t')
        if not p:
            continue
        candidate = (buf + ' ' + p).strip() if buf else p
        if len(candidate) <= limit:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            # frase longa demais: quebra dura
            while len(p) > limit:
                cut = p.rfind(' ', 0, limit)
                cut = cut if cut > 40 else limit
                chunks.append(p[:cut].strip())
                p = p[cut:].strip()
            buf = p
    if buf:
        chunks.append(buf)
    return chunks


class TTSService:
    """Singleton Piper. Lazy: só carrega/baixa voz no primeiro uso."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self):
        self._synth_lock = threading.Lock()
        self._voice_cache = {}  # onnx_path -> PiperVoice (lib python)
        VOICE_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── vozes ──
    def available_voices(self) -> list:
        voices = []
        if self._edge_tts_available():
            voices.extend(list(EDGE_VOICES.keys()))
        found_piper = sorted(p.stem.replace('.onnx', '') for p in VOICE_DIR.glob('*.onnx'))
        for name in VOICE_CATALOG:
            if name not in found_piper:
                found_piper.append(name)
        voices.extend(found_piper)
        if DEFAULT_VOICE not in voices:
            voices.insert(0, DEFAULT_VOICE)
        return voices

    def voice_files(self, voice: str) -> tuple:
        onnx = VOICE_DIR / f'{voice}.onnx'
        cfg = VOICE_DIR / f'{voice}.onnx.json'
        return onnx, cfg

    def is_voice_downloaded(self, voice: str) -> bool:
        if voice in EDGE_VOICES or voice.startswith('pt-BR-'):
            return self._edge_tts_available()
        onnx, cfg = self.voice_files(voice)
        return onnx.exists() and cfg.exists()

    def ensure_voice(self, voice: str) -> tuple:
        onnx, cfg = self.voice_files(voice)
        if onnx.exists() and cfg.exists():
            return onnx, cfg
        if voice not in VOICE_CATALOG:
            raise FileNotFoundError(
                f"Voz '{voice}' não encontrada em {VOICE_DIR}. "
                f"Vozes conhecidas: {', '.join(VOICE_CATALOG)}."
            )
        url_onnx, url_json = VOICE_CATALOG[voice]
        logger.info('Baixando voz Piper %s ...', voice)
        try:
            self._download(url_onnx, onnx)
            self._download(url_json, cfg)
        except Exception as e:
            for f in (onnx, cfg):
                try:
                    if f.exists() and f.stat().st_size == 0:
                        f.unlink()
                except OSError:
                    pass
            raise RuntimeError(f'Falha ao baixar voz {voice}: {e}')
        return onnx, cfg

    @staticmethod
    def _download(url: str, dest: Path):
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + '.part')
        req = urllib.request.Request(url, headers={'User-Agent': 'seco-live/1.0'})
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, 'wb') as f:
            shutil.copyfileobj(r, f)
        tmp.replace(dest)

    # ── detecção dos motores ──
    @staticmethod
    def _edge_tts_available() -> bool:
        try:
            import edge_tts  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _python_piper_available() -> bool:
        try:
            import piper  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _cli_available() -> bool:
        return shutil.which('piper') is not None

    def status(self) -> dict:
        edge_ok = self._edge_tts_available()
        piper_ok = self._python_piper_available() or self._cli_available()
        return {
            'engine': 'edge-tts' if edge_ok else 'piper',
            'edge_tts': edge_ok,
            'python_lib': self._python_piper_available(),
            'cli': self._cli_available(),
            'available': edge_ok or piper_ok,
            'default_voice': DEFAULT_VOICE,
            'default_voice_downloaded': self.is_voice_downloaded(DEFAULT_VOICE),
            'voices': self.available_voices(),
            'max_chars_per_request': MAX_CHARS,
            'default_noise_scale': DEFAULT_NOISE_SCALE,
            'default_noise_w': DEFAULT_NOISE_W,
        }

    # ── síntese ──
    def synthesize(self, text: str, voice: str = None, speed: float = 1.0,
                   noise_scale: float = None, noise_w: float = None) -> Path:
        clean = clean_text_for_speech(text)
        if not clean:
            raise ValueError('Texto vazio após limpeza — nada para falar.')
        if len(clean) > MAX_CHARS:
            raise ValueError(f'Texto com {len(clean)} caracteres (limite {MAX_CHARS}). Divida em frases.')
        voice = (voice or DEFAULT_VOICE).strip() or DEFAULT_VOICE
        voice = VOICE_ALIASES.get(voice.lower(), voice)
        speed = max(0.5, min(2.0, float(speed or 1.0)))

        # 1) Tenta sintetizar com Edge-TTS (voz neural ultra-humana)
        is_edge = voice.startswith('pt-BR-') or voice in EDGE_VOICES
        if is_edge and self._edge_tts_available():
            edge_voice = voice if voice in EDGE_VOICES else 'pt-BR-AntonioNeural'
            key = hashlib.sha1(f'edge_v2|{edge_voice}|{speed:.2f}|{clean}'.encode('utf-8')).hexdigest()
            out = CACHE_DIR / f'{key}.mp3'
            if out.exists() and out.stat().st_size > 100:
                return out
            try:
                self._synth_edge(edge_voice, clean, out, speed)
                if out.exists() and out.stat().st_size > 100:
                    return out
            except Exception as e:
                logger.warning('Edge-TTS falhou (%s), tentando fallback para Piper: %s', edge_voice, e)

        # 2) Fallback ou modo offline via Piper local
        piper_voice = voice if voice in VOICE_CATALOG else getattr(settings, 'PIPER_VOICE', 'pt_BR-faber-medium')
        ns = max(0.0, min(2.0, float(DEFAULT_NOISE_SCALE if noise_scale is None else noise_scale)))
        nw = max(0.0, min(2.0, float(DEFAULT_NOISE_W if noise_w is None else noise_w)))

        key = hashlib.sha1(f'v2|{piper_voice}|{speed:.2f}|{ns:.2f}|{nw:.2f}|{clean}'.encode('utf-8')).hexdigest()
        out = CACHE_DIR / f'{key}.wav'
        if out.exists() and out.stat().st_size > 44:
            return out

        onnx, cfg = self.ensure_voice(piper_voice)

        with self._synth_lock:  # Piper não é thread-safe; serializa
            if out.exists() and out.stat().st_size > 44:
                return out
            if self._python_piper_available():
                try:
                    self._synth_python(onnx, cfg, clean, out, speed, ns, nw)
                    return out
                except Exception as e:
                    logger.warning('Piper lib falhou, tentando CLI: %s', e)
            if self._cli_available():
                self._synth_cli(onnx, cfg, clean, out, speed, ns, nw)
                return out
        raise PiperNotAvailable(
            'Nenhum motor de voz disponível (Edge-TTS e Piper falharam).'
        )

    def _synth_edge(self, voice: str, text: str, out: Path, speed: float):
        import edge_tts
        pct = int(round((speed - 1.0) * 100))
        rate_str = f"{pct:+d}%"

        async def _run():
            comm = edge_tts.Communicate(text, voice, rate=rate_str)
            await comm.save(str(out))

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(lambda: asyncio.run(_run())).result()

    def _synth_python(self, onnx: Path, cfg: Path, text: str, out: Path,
                      speed: float, noise_scale: float, noise_w: float):
        from piper import PiperVoice
        key = str(onnx)
        voice = self._voice_cache.get(key)
        if voice is None:
            voice = PiperVoice.load(str(onnx), config_path=str(cfg))
            self._voice_cache[key] = voice
        length_scale = 1.0 / speed
        chunks = []
        sample_rate = 22050
        # API piper-tts: synthesize(text, ...) -> generator de AudioChunk.
        # Assinatura varia entre versões; tenta do mais completo ao básico.
        # (piper>=1.2 usa SynthesisConfig; versões antigas aceitam kwargs.)
        audio_iter = None
        try:
            from piper.config import SynthesisConfig
            audio_iter = voice.synthesize(
                text,
                syn_config=SynthesisConfig(
                    length_scale=length_scale,
                    noise_scale=noise_scale,
                    noise_w_scale=noise_w,
                ),
            )
        except (ImportError, TypeError):
            audio_iter = None
        if audio_iter is None:
            try:
                audio_iter = voice.synthesize(text, length_scale=length_scale,
                                              noise_scale=noise_scale, noise_w=noise_w)
            except TypeError:
                try:
                    audio_iter = voice.synthesize(text, length_scale=length_scale)
                except TypeError:
                    audio_iter = voice.synthesize(text)
        for chunk in audio_iter:
            sample_rate = getattr(chunk, 'sample_rate', sample_rate)
            data = getattr(chunk, 'audio_int16_bytes', b'')
            if data:
                chunks.append(data)
        if not chunks:
            raise RuntimeError('Piper retornou áudio vazio.')
        with wave.open(str(out), 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            for c in chunks:
                w.writeframes(c)

    def _synth_cli(self, onnx: Path, cfg: Path, text: str, out: Path,
                   speed: float, noise_scale: float, noise_w: float):
        import subprocess
        length_scale = 1.0 / speed
        cmd = [
            'piper',
            '--model', str(onnx),
            '--config', str(cfg),
            '--output_file', str(out),
            '--length_scale', str(length_scale),
            '--noise_scale', str(noise_scale),
            '--noise_w', str(noise_w),
        ]
        try:
            subprocess.run(cmd, input=text.encode('utf-8'), timeout=180, check=True,
                           capture_output=True)
        except subprocess.CalledProcessError as e:
            err = (e.stderr or b'').decode('utf-8', 'ignore')[-1000:]
            raise RuntimeError(f'Piper CLI falhou: {err}')
        if not out.exists() or out.stat().st_size <= 44:
            raise RuntimeError('Piper CLI não gerou áudio.')


tts_service = TTSService()
