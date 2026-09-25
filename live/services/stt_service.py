"""Serviço STT (Faster-Whisper) do modo Live — singleton isolado.

Transcrição 100% local (CPU). O modelo baixa sozinho no primeiro uso
(~500MB no 'small'). Falhas aqui nunca afetam o chat.
"""
import logging
import threading
from pathlib import Path

from django.conf import settings

logger = logging.getLogger(__name__)

BASE_DIR = Path(settings.BASE_DIR)

MODEL_NAME = getattr(settings, 'LIVE_STT_MODEL', 'small')  # tiny|base|small|medium
MODEL_DIR = Path(getattr(settings, 'LIVE_STT_DIR', BASE_DIR / 'models' / 'stt'))
DEVICE = getattr(settings, 'LIVE_STT_DEVICE', 'cpu')
COMPUTE = getattr(settings, 'LIVE_STT_COMPUTE', 'int8')


class STTEngineMissing(Exception):
    pass


class STTService:
    """Singleton Faster-Whisper. Lazy: só carrega no primeiro uso."""

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
        self._model = None
        self._model_lock = threading.Lock()
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _engine_available() -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def status(self) -> dict:
        return {
            'engine': 'faster-whisper',
            'available': self._engine_available(),
            'model': MODEL_NAME,
            'loaded': self._model is not None,
            'device': DEVICE,
        }

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        if not self._engine_available():
            raise STTEngineMissing(
                'Transcrição local indisponível. Instale com: '
                'pip install faster-whisper. O chat continua funcionando normalmente.'
            )
        from faster_whisper import WhisperModel
        logger.info('Carregando Whisper %s (primeiro uso pode baixar o modelo)...', MODEL_NAME)
        self._model = WhisperModel(
            MODEL_NAME, device=DEVICE, compute_type=COMPUTE, download_root=str(MODEL_DIR)
        )
        return self._model

    def transcribe(self, audio_path: str, language: str = 'pt') -> str:
        """Transcreve um arquivo de áudio (wav/webm/mp3/ogg) para texto."""
        with self._model_lock:  # serializa como no TTS
            model = self._ensure_model()
            segments, _info = model.transcribe(
                audio_path, language=language, beam_size=5, vad_filter=True,
            )
            text = ''.join(s.text for s in segments).strip()
        if not text:
            raise ValueError('Não reconheci fala no áudio. Tente falar mais perto do microfone.')
        return text


stt_service = STTService()
