"""Endpoints do modo Live — prefixo /api/live/. Nada aqui toca no chat."""
import logging

from django.http import FileResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status

from .services.tts_service import tts_service, PiperNotAvailable, split_into_chunks

logger = logging.getLogger(__name__)


@api_view(['GET'])
@permission_classes([AllowAny])
def live_health(request):
    """Estado do motor de voz (para o botão Live saber se pode ativar)."""
    try:
        return Response({'ok': True, 'live': tts_service.status()})
    except Exception as e:
        logger.exception('live_health falhou')
        return Response({'ok': False, 'error': str(e)},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([AllowAny])
def live_voices(request):
    try:
        s = tts_service.status()
        return Response({
            'default': s['default_voice'],
            'voices': [
                {'id': v, 'downloaded': tts_service.is_voice_downloaded(v)}
                for v in s['voices']
            ],
        })
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@permission_classes([AllowAny])
def live_speak(request):
    """Texto -> WAV. Uma frase por chamada (frontend divide a resposta).

    Body: {"text": "...", "voice": "pt_BR-faber-medium" (opcional), "speed": 0.9 (opcional),
           "noise_scale": 0.8 (opcional), "noise_w": 0.8 (opcional)}
    """
    data = request.data or {}
    text = (data.get('text') or '').strip()
    voice = (data.get('voice') or '').strip() or None

    def _f(key, default):
        try:
            v = data.get(key, default)
            return float(default if v is None or v == '' else v)
        except (TypeError, ValueError):
            return default

    speed = _f('speed', 0.9)
    noise_scale = data.get('noise_scale', None)
    noise_w = data.get('noise_w', None)
    try:
        noise_scale = None if noise_scale in (None, '') else float(noise_scale)
        noise_w = None if noise_w in (None, '') else float(noise_w)
    except (TypeError, ValueError):
        return Response({'error': 'noise_scale/noise_w inválidos.'},
                        status=status.HTTP_400_BAD_REQUEST)

    if not text:
        return Response({'error': 'Campo "text" vazio.'},
                        status=status.HTTP_400_BAD_REQUEST)
    try:
        wav = tts_service.synthesize(text, voice=voice, speed=speed,
                                     noise_scale=noise_scale, noise_w=noise_w)
    except ValueError as e:
        return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
    except FileNotFoundError as e:
        return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
    except PiperNotAvailable as e:
        return Response({'error': str(e), 'install_hint': True},
                        status=status.HTTP_503_SERVICE_UNAVAILABLE)
    except RuntimeError as e:
        logger.exception('live_speak falhou')
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except Exception:
        logger.exception('live_speak falhou')
        return Response({'error': 'Falha interna ao sintetizar voz.'},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    resp = FileResponse(open(wav, 'rb'), content_type='audio/wav')
    resp['Content-Length'] = wav.stat().st_size
    resp['Cache-Control'] = 'public, max-age=86400'
    resp['X-Live-Voice'] = voice or tts_service.status()['default_voice']
    return resp


@api_view(['POST'])
@permission_classes([AllowAny])
def live_split(request):
    """Divide uma resposta longa em frases curtas (limpas p/ fala). Poupa o frontend."""
    data = request.data or {}
    text = data.get('text') or ''
    try:
        limit = int(data.get('limit', 220) or 220)
    except (TypeError, ValueError):
        limit = 220
    limit = max(80, min(600, limit))
    try:
        return Response({'chunks': split_into_chunks(text, limit=limit)})
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
