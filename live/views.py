"""Endpoints do modo Live — prefixo /api/live/. Nada aqui toca no chat."""
import logging

from django.http import FileResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status

from .services.tts_service import (
    tts_service, PiperNotAvailable, split_into_chunks,
    EDGE_VOICES, VOICE_CATALOG,
)
from .services.stt_service import stt_service, STTEngineMissing

logger = logging.getLogger(__name__)

MAX_AUDIO_MB = 25


@api_view(['GET'])
@permission_classes([AllowAny])
def live_health(request):
    """Estado dos motores de voz (para o botão Live saber se pode ativar)."""
    try:
        return Response({'ok': True,
                         'live': tts_service.status(),
                         'stt': stt_service.status()})
    except Exception as e:
        logger.exception('live_health falhou')
        return Response({'ok': False, 'error': str(e)},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([AllowAny])
def live_voices(request):
    try:
        s = tts_service.status()
        voices_list = []
        # Vozes neurais Edge-TTS
        if tts_service._edge_tts_available():
            for vid, meta in EDGE_VOICES.items():
                voices_list.append({
                    'id': vid,
                    'name': meta['name'],
                    'engine': 'edge-tts',
                    'gender': meta['gender'],
                    'downloaded': True,
                })
        # Vozes locais Piper
        for vid in VOICE_CATALOG:
            voices_list.append({
                'id': vid,
                'name': f"{vid} (Piper Local)",
                'engine': 'piper',
                'gender': 'male',
                'downloaded': tts_service.is_voice_downloaded(vid),
            })

        return Response({
            'default': s['default_voice'],
            'voices': voices_list,
        })
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@permission_classes([AllowAny])
def live_speak(request):
    """Texto -> Áudio (MP3/WAV). Uma frase por chamada (frontend divide a resposta).

    Body: {"text": "...", "voice": "pt-BR-AntonioNeural" (opcional), "speed": 0.9 (opcional),
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

    speed = _f('speed', 0.95)
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
        audio_file = tts_service.synthesize(text, voice=voice, speed=speed,
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

    content_type = 'audio/mpeg' if str(audio_file).endswith('.mp3') else 'audio/wav'
    resp = FileResponse(open(audio_file, 'rb'), content_type=content_type)
    resp['Content-Length'] = audio_file.stat().st_size
    resp['Cache-Control'] = 'public, max-age=86400'
    resp['X-Live-Voice'] = voice or tts_service.status()['default_voice']
    return resp


@api_view(['POST'])
@permission_classes([AllowAny])
def live_transcribe(request):
    """Áudio do microfone -> texto (Faster-Whisper local).

    Multipart: {"audio": arquivo (webm/wav/mp3/ogg, até 25MB), "language": "pt" (opcional)}
    """
    f = request.FILES.get('audio')
    if f is None:
        return Response({'error': 'Nenhum áudio enviado (campo "audio").'},
                        status=status.HTTP_400_BAD_REQUEST)
    if f.size > MAX_AUDIO_MB * 1024 * 1024:
        return Response({'error': f'Áudio maior que {MAX_AUDIO_MB}MB.'},
                        status=status.HTTP_400_BAD_REQUEST)
    language = (request.data.get('language') or 'pt').strip() or 'pt'

    import tempfile, os
    suffix = os.path.splitext(getattr(f, 'name', '') or '')[1].lower() or '.webm'
    # Formatos de gravadores mobile (Flutter record / iOS / Android).
    # Mantém a extensão original quando desconhecida: o decodificador
    # (ffmpeg/PyAV) fareja o conteúdo pelo cabeçalho, não pela extensão.
    _KNOWN_AUDIO = ('.webm', '.wav', '.mp3', '.ogg', '.oga', '.opus',
                    '.m4a', '.mp4', '.aac', '.caf', '.3gp', '.flac')
    if suffix not in _KNOWN_AUDIO and suffix:
        logger.warning('live_transcribe: extensão incomum %s, mantendo original', suffix)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        for chunk in f.chunks():
            tmp.write(chunk)
        tmp.close()
        try:
            text = stt_service.transcribe(tmp.name, language=language)
        except STTEngineMissing as e:
            return Response({'error': str(e), 'install_hint': True},
                            status=status.HTTP_503_SERVICE_UNAVAILABLE)
        except ValueError as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        return Response({'text': text})
    except Exception:
        logger.exception('live_transcribe falhou')
        return Response({'error': 'Falha interna ao transcrever áudio.'},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@api_view(['POST'])
@permission_classes([AllowAny])
def live_turn(request):
    """Turno completo do modo Live para app mobile — 1 chamada em vez de 5.

    Multipart (gravador do celular):
      audio: arquivo (m4a/wav/mp3/ogg/webm/aac..., até 25MB)
      language: "pt" (opcional) | voice: id da voz (opcional)
      speed: 0.95 (opcional) | history: JSON list (opcional)
      system_prompt: texto extra (opcional — o brevity do Live é injetado sozinho)
      max_tokens: 256 (opcional) | temperature: 0.2 (opcional)
      use_remote: true/false (opcional, default = config do servidor)
      use_rag: true/false (opcional, default false — mais rápido no mobile)
      include_audio: true/false (opcional, default true)
      max_chunks: 1-12 (opcional, default 6 — limita tamanho/latência)

    JSON (sem mic, p/ teste):
      {"text": "...", ...mesmos opcionais}

    Retorno:
      {"text": transcricao, "resposta": "...", "chunks": [...],
       "audios": [{"index": 0, "text": frase, "mime": "audio/mpeg",
                   "audio_base64": "...", "voice": "..."}],
       "voice": "...", "speed": 0.95}
    """
    import base64
    import json as _json

    data = request.data or {}

    def _f(key, default):
        try:
            v = data.get(key, default)
            return float(default if v is None or v == '' else v)
        except (TypeError, ValueError):
            return default

    def _b(key, default):
        v = data.get(key, default)
        if isinstance(v, bool):
            return v
        if v is None or v == '':
            return default
        return str(v).strip().lower() in ('1', 'true', 'sim', 'yes', 'y')

    def _i(key, default, lo, hi):
        try:
            v = int(data.get(key, default))
        except (TypeError, ValueError):
            v = default
        return max(lo, min(hi, v))

    language = (data.get('language') or 'pt')
    language = str(language).strip() or 'pt'
    voice = (data.get('voice') or '').strip() if isinstance(data.get('voice'), str) else None
    voice = voice or None
    speed = max(0.5, min(2.0, _f('speed', 0.95)))
    temperature = max(0.0, min(1.0, _f('temperature', 0.2)))
    max_tokens = _i('max_tokens', 256, 32, 1024)
    max_chunks = _i('max_chunks', 6, 1, 12)
    include_audio = _b('include_audio', True)
    use_rag = _b('use_rag', False)

    use_remote = data.get('use_remote', None)
    if isinstance(use_remote, str):
        use_remote = use_remote.strip().lower() in ('1', 'true', 'sim', 'yes', 'y')
    remote_config = data.get('remote_config', {}) or {}
    if isinstance(remote_config, str):
        try:
            remote_config = _json.loads(remote_config)
        except (TypeError, ValueError):
            remote_config = {}

    history = data.get('history', []) or []
    if isinstance(history, str):
        try:
            history = _json.loads(history)
        except (TypeError, ValueError):
            history = []
    if not isinstance(history, list):
        history = []

    user_system = (data.get('system_prompt') or '')
    user_system = user_system if isinstance(user_system, str) else ''
    if 'MODO LIVE' not in user_system:
        system_prompt = (
            '[MODO LIVE: seja objetivo e direto. '
            'Responda em no máximo 4 frases curtas. '
            'PROIBIDO listas, código, saudações longas e perguntas de volta.]'
            + (f'\n\n{user_system}' if user_system.strip() else '')
        )
    else:
        system_prompt = user_system

    # 1) Áudio -> texto (ou texto direto no modo JSON)
    text = ''
    f = request.FILES.get('audio')
    if f is not None:
        if f.size > MAX_AUDIO_MB * 1024 * 1024:
            return Response({'error': f'Áudio maior que {MAX_AUDIO_MB}MB.'},
                            status=status.HTTP_400_BAD_REQUEST)
        import tempfile, os
        suffix = os.path.splitext(getattr(f, 'name', '') or '')[1].lower() or '.m4a'
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            for chunk in f.chunks():
                tmp.write(chunk)
            tmp.close()
            try:
                text = stt_service.transcribe(tmp.name, language=language)
            except STTEngineMissing as e:
                return Response({'error': str(e), 'install_hint': True},
                                status=status.HTTP_503_SERVICE_UNAVAILABLE)
            except ValueError as e:
                return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception:
            logger.exception('live_turn: transcrição falhou')
            return Response({'error': 'Falha interna ao transcrever áudio.'},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
    else:
        text = ((data.get('text') or '') if isinstance(data.get('text'), str) else '').strip()
        if not text:
            return Response(
                {'error': 'Envie o campo "audio" (multipart) ou "text" (JSON).'},
                status=status.HTTP_400_BAD_REQUEST)

    # 2) Texto -> resposta da IA (reusa os singletons do chat; não recarrega o modelo)
    try:
        from api.views import lmm_service, remote_llm_service
    except Exception as e:
        logger.exception('live_turn: serviços de chat indisponíveis')
        return Response({'error': f'Serviço de chat indisponível: {e}'},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    if remote_config:
        try:
            remote_llm_service.set_config(remote_config)
        except Exception:
            pass
    if use_remote is True:
        if not remote_llm_service.is_enabled():
            return Response(
                {'error': 'Modelo remoto não configurado.'},
                status=status.HTTP_400_BAD_REQUEST)
        service = remote_llm_service
    elif use_remote is False:
        service = lmm_service
    else:
        try:
            service = remote_llm_service if remote_llm_service.is_enabled() else lmm_service
        except Exception:
            service = lmm_service
    try:
        resposta = service.generate_response(
            prompt=text, temperature=temperature, image_base64=None,
            system_prompt=system_prompt, history=history,
            use_rag=use_rag, max_tokens=max_tokens)
    except Exception:
        logger.exception('live_turn: generate_response falhou')
        return Response({'error': 'Falha ao gerar resposta da IA.'},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    if not (resposta or '').strip():
        return Response({'error': 'A IA retornou resposta vazia.'},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # 3) Divide em frases curtas p/ fala
    try:
        chunks = split_into_chunks(resposta, limit=220)[:max_chunks]
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # 4) Frases -> áudios (base64 p/ o app tocar sem novas chamadas)
    audios = []
    voice_used = voice or tts_service.status().get('default_voice')
    if include_audio:
        for idx, frase in enumerate(chunks):
            try:
                audio_file = tts_service.synthesize(frase, voice=voice, speed=speed)
                raw = audio_file.read_bytes()
                if not raw:
                    audios.append({'index': idx, 'text': frase, 'error': 'Áudio vazio.'})
                    continue
                mime = 'audio/mpeg' if str(audio_file).endswith('.mp3') else 'audio/wav'
                audios.append({
                    'index': idx,
                    'text': frase,
                    'mime': mime,
                    'audio_base64': base64.b64encode(raw).decode('ascii'),
                    'voice': voice_used,
                })
            except (ValueError, FileNotFoundError) as e:
                audios.append({'index': idx, 'text': frase, 'error': str(e)})
            except PiperNotAvailable as e:
                return Response({'error': str(e), 'install_hint': True},
                                status=status.HTTP_503_SERVICE_UNAVAILABLE)
            except Exception:
                logger.exception('live_turn: TTS falhou no chunk %s', idx)
                audios.append({'index': idx, 'text': frase, 'error': 'Falha ao sintetizar.'})

    payload = {
        'text': text,
        'resposta': resposta,
        'chunks': chunks,
        'audios': audios,
        'voice': voice_used,
        'speed': speed,
    }
    if include_audio and not audios:
        payload['warning'] = 'Nenhum áudio gerado; exiba ao menos o texto.'
    return Response(payload)


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
