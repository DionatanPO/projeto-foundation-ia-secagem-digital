from rest_framework.decorators import api_view, permission_classes, authentication_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from django.http import StreamingHttpResponse, JsonResponse
import logging
import json
import psutil
import os
import re
from .serializers import ChatRequestSerializer, ChatResponseSerializer, ModelSwitchSerializer

logger = logging.getLogger(__name__)
from .services.lmm_service import LMMService
from .services.remote_llm_service import RemoteLLMService
from .services.attachment_service import extract_attachments_text

# Instancia os serviços
lmm_service = LMMService()
remote_llm_service = RemoteLLMService()

@api_view(['GET'])
@permission_classes([AllowAny])
def health_check(request):
    return Response({"status": "ok", "message": "Django API is running"})

@api_view(['GET'])
def system_status(request):
    """
    Retorna o consumo de memória RAM do processo atual (Django + LMM) e do sistema,
    mais o estado real de GPU/CPU do modelo carregado.
    """
    process = psutil.Process(os.getpid())
    process_memory_mb = process.memory_info().rss / (1024 * 1024)
    system_memory = psutil.virtual_memory()
    total_memory_mb = system_memory.total / (1024 * 1024)
    used_memory_mb = system_memory.used / (1024 * 1024)
    memory_percent = system_memory.percent

    try:
        device_info = lmm_service.get_device_info()
    except Exception as e:
        device_info = {"gpu_enabled": False, "device_requested": "cpu", "n_gpu_layers": 0, "error": str(e)}

    return Response({
        "process_ram_mb": round(process_memory_mb, 2),
        "system_total_mb": round(total_memory_mb, 2),
        "system_used_mb": round(used_memory_mb, 2),
        "system_percent": memory_percent,
        "gpu_enabled": device_info.get("gpu_enabled", False),
        "device": device_info.get("device_requested", "cpu"),
        "n_gpu_layers": device_info.get("n_gpu_layers", 0),
        "gpu_backend_available": device_info.get("gpu_backend_available", False),
        "backend_info": device_info.get("backend_info", ""),
    })

@api_view(['POST'])
def chat_inference(request):
    """
    Endpoint para enviar um prompt ao modelo LMM e obter a resposta estruturada.
    Suporta modelo local e remoto.
    """
    serializer = ChatRequestSerializer(data=request.data)

    if serializer.is_valid():
        prompt = serializer.validated_data['prompt']
        temperature = max(0.0, min(1.0, serializer.validated_data['temperature']))
        image_base64 = serializer.validated_data.get('image_base64', None)
        system_prompt = serializer.validated_data.get('system_prompt', None)
        history = serializer.validated_data.get('history', [])
        use_rag = serializer.validated_data.get('use_rag', True)
        use_remote = serializer.validated_data.get('use_remote', None)
        remote_config = serializer.validated_data.get('remote_config', {})
        max_tokens = serializer.validated_data.get('max_tokens', None)
        if max_tokens is not None:
            max_tokens = max(32, min(4096, int(max_tokens)))
        attachment_ctx = extract_attachments_text(serializer.validated_data.get('attachments', []))
        if attachment_ctx:
            prompt = f"{attachment_ctx}\n\nPERGUNTA DO USUÁRIO: {prompt}"

        if remote_config:
            remote_llm_service.set_config(remote_config)

        if use_remote is True:
            if not remote_llm_service.is_enabled():
                return Response(
                    {"error": "Modelo remoto não configurado. Verifique a URL da API e se o toggle está ativo."},
                    status=status.HTTP_400_BAD_REQUEST
                )
            service = remote_llm_service
        elif use_remote is False:
            service = lmm_service
        else:
            service = remote_llm_service if remote_llm_service.is_enabled() else lmm_service

        remote_cfg = remote_llm_service.get_config()
        logger.info(
            "chat_inference | use_remote=%s | service=%s | remote_cfg=%s | remote_enabled=%s",
            use_remote,
            "remote" if service == remote_llm_service else "local",
            remote_cfg.get("api_url", "n/a"),
            remote_cfg.get("enabled", False)
        )

        answer_text = service.generate_response(
            prompt=prompt,
            temperature=temperature,
            image_base64=image_base64,
            system_prompt=system_prompt,
            history=history,
            use_rag=use_rag,
            max_tokens=max_tokens
        )

        return Response({
            'resposta': answer_text,
            'thinking': ""
        }, status=status.HTTP_200_OK)

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
def chat_stream(request):
    """
    Endpoint de streaming para resposta em tempo real.
    Suporta modelo local (LMM) e modelo remoto (OpenAI-compatible).
    """
    serializer = ChatRequestSerializer(data=request.data)
    if serializer.is_valid():
        prompt = serializer.validated_data['prompt']
        temperature = serializer.validated_data.get('temperature', 0.2)
        image_base64 = serializer.validated_data.get('image_base64', None)
        system_prompt = serializer.validated_data.get('system_prompt', None)
        history = serializer.validated_data.get('history', [])
        use_rag = serializer.validated_data.get('use_rag', True)
        use_remote = serializer.validated_data.get('use_remote', None)
        remote_config = serializer.validated_data.get('remote_config', {})
        max_tokens = serializer.validated_data.get('max_tokens', None)
        if max_tokens is not None:
            max_tokens = max(32, min(4096, int(max_tokens)))
        attachment_ctx = extract_attachments_text(serializer.validated_data.get('attachments', []))
        if attachment_ctx:
            prompt = f"{attachment_ctx}\n\nPERGUNTA DO USUÁRIO: {prompt}"

        if remote_config:
            remote_llm_service.set_config(remote_config)

        if use_remote is True:
            if not remote_llm_service.is_enabled():
                return Response(
                    {"error": "Modelo remoto não configurado. Verifique a URL da API e se o toggle está ativo."},
                    status=status.HTTP_400_BAD_REQUEST
                )
            use_remote_flag = True
        elif use_remote is False:
            use_remote_flag = False
        else:
            use_remote_flag = remote_llm_service.is_enabled()

        remote_cfg = remote_llm_service.get_config()
        logger.info(
            "chat_stream | use_remote=%s | use_remote_flag=%s | remote_cfg=%s | remote_enabled=%s",
            use_remote, use_remote_flag,
            remote_cfg.get("api_url", "n/a"),
            remote_cfg.get("enabled", False)
        )

        def stream_generator():
            try:
                service = remote_llm_service if use_remote_flag else lmm_service
                for chunk in service.generate_stream(
                    prompt=prompt,
                    temperature=temperature,
                    image_base64=image_base64,
                    system_prompt=system_prompt,
                    history=history,
                    use_rag=use_rag,
                    max_tokens=max_tokens
                ):
                    yield chunk
            except (BrokenPipeError, ConnectionResetError):
                pass

        return StreamingHttpResponse(stream_generator(), content_type='text/plain')

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET'])
@permission_classes([AllowAny])
def service_mode(request):
    """
    Retorna o modo atual (remoto ou local) baseado na config salva.
    Útil para o Flutter verificar antes de enviar mensagens.
    """
    cfg = remote_llm_service.get_config()
    enabled = remote_llm_service.is_enabled()
    try:
        device_info = lmm_service.get_device_info()
    except Exception:
        device_info = {}
    return Response({
        "mode": "remote" if enabled else "local",
        "remote_enabled": enabled,
        "remote_api_url": cfg.get("api_url", ""),
        "local_model_loaded": lmm_service.model is not None,
        "local_model": lmm_service.get_current_model(),
        "device": device_info.get("device_requested", "cpu"),
        "gpu_enabled": device_info.get("gpu_enabled", False),
    })



@api_view(['POST'])
def remote_config_save(request):
    """
    Salva a configuração do modelo remoto no servidor.
    """
    config = request.data.get('config', {})
    remote_llm_service.set_config(config)
    return Response({"status": "Configuração salva"})


@api_view(['GET'])
def remote_config_load(request):
    """
    Retorna a configuração atual do modelo remoto.
    """
    return Response(remote_llm_service.get_config())


@api_view(['POST'])
def remote_config_test(request):
    """
    Testa a conexão com a API remota.
    """
    config = request.data.get('config', {})
    remote_llm_service.set_config(config)
    result = remote_llm_service.test_connection()
    return Response(result)

@api_view(['GET'])
def list_models(request):
    """
    Lista os modelos disponíveis na pasta models.
    """
    models = lmm_service.list_available_models()
    current_model = lmm_service.get_current_model()
    try:
        device_info = lmm_service.get_device_info()
    except Exception:
        device_info = {}
    return Response({
        "models": models,
        "current_model": current_model,
        "device": device_info.get("device_requested", "cpu"),
        "gpu_enabled": device_info.get("gpu_enabled", False),
        "n_gpu_layers": device_info.get("n_gpu_layers", 0),
    })

@api_view(['POST'])
def switch_model(request):
    """
    Troca o modelo carregado.
    """
    serializer = ModelSwitchSerializer(data=request.data)
    if serializer.is_valid():
        model_name = serializer.validated_data['model_name']
        use_gpu = serializer.validated_data.get('use_gpu', False)
        success = lmm_service.switch_model(model_name, use_gpu=use_gpu)
        if success:
            try:
                device_info = lmm_service.get_device_info()
            except Exception:
                device_info = {}
            resp = {"status": "Model switched", "model": model_name}
            resp.update({
                "device": device_info.get("device_requested", "gpu" if use_gpu else "cpu"),
                "gpu_enabled": device_info.get("gpu_enabled", False),
                "n_gpu_layers": device_info.get("n_gpu_layers", 0),
            })
            if use_gpu and not resp["gpu_enabled"]:
                resp["warning"] = (
                    "GPU solicitada, mas o build do llama-cpp-python é CPU-only "
                    "(sem VULKAN/CUDA). Modelo carregado em CPU. "
                    "Reinstale com: CMAKE_ARGS='-DGGML_VULKAN=on' pip install --force-reinstall --no-cache-dir llama-cpp-python"
                )
            return Response(resp)
        return Response({"error": "Failed to load model"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def clear_rag_storage(request):
    """
    Endpoint para apagar e reconstruir o índice RAG a partir dos documentos atuais em /documents.
    """
    from .services.rag_service import RagService
    rag = RagService()
    success = rag.clear_and_rebuild_storage()
    if success:
        return Response({"status": "RAG storage cleared and rebuilt successfully!"}, status=status.HTTP_200_OK)
    return Response({"error": "Failed to clear/rebuild RAG storage"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# ── Documentos da base de conhecimento (RAG) ──
RAG_ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.txt', '.md', '.csv'}
RAG_MAX_FILE_MB = 50
RAG_MAX_FILES_PER_REQUEST = 20


def _rag_documents_dir():
    from django.conf import settings as dj_settings
    docs_dir = os.path.join(dj_settings.BASE_DIR, 'documents')
    os.makedirs(docs_dir, exist_ok=True)
    return docs_dir


def _sanitize_doc_filename(raw):
    name = os.path.basename((raw or '').strip())
    name = re.sub(r'[^\w.\- ]', '_', name, flags=re.UNICODE)
    name = name.strip(' .')
    return name[:200] if name else ''


def _list_rag_documents():
    docs_dir = _rag_documents_dir()
    items = []
    for entry in sorted(os.listdir(docs_dir)):
        full = os.path.join(docs_dir, entry)
        if os.path.isfile(full):
            items.append({
                "name": entry,
                "size_kb": round(os.path.getsize(full) / 1024, 1),
            })
    return items


@api_view(['GET'])
def rag_documents_list(request):
    """
    Lista os arquivos da pasta documents/ usados pelo RAG.
    """
    return Response({"documents": _list_rag_documents()})


@api_view(['POST'])
def rag_documents_upload(request):
    """
    Recebe arquivos (multipart, campo 'files'), salva na pasta documents/
    e reconstrói o índice RAG para que entrem em vigor imediatamente.
    """
    from .services.rag_service import RagService

    files = request.FILES.getlist('files')
    if not files:
        return Response(
            {"error": "Nenhum arquivo enviado. Use o campo 'files' (multipart)."},
            status=status.HTTP_400_BAD_REQUEST,
        )
    if len(files) > RAG_MAX_FILES_PER_REQUEST:
        return Response(
            {"error": f"Máximo de {RAG_MAX_FILES_PER_REQUEST} arquivos por envio."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    docs_dir = _rag_documents_dir()
    saved, skipped = [], []
    for f in files:
        safe_name = _sanitize_doc_filename(getattr(f, 'name', ''))
        ext = os.path.splitext(safe_name)[1].lower()
        if not safe_name or ext not in RAG_ALLOWED_EXTENSIONS:
            skipped.append({"name": getattr(f, 'name', '?'),
                            "reason": f"Extensão não suportada. Use: {', '.join(sorted(RAG_ALLOWED_EXTENSIONS))}"})
            continue
        if f.size is not None and f.size > RAG_MAX_FILE_MB * 1024 * 1024:
            skipped.append({"name": safe_name,
                            "reason": f"Arquivo maior que {RAG_MAX_FILE_MB} MB."})
            continue
        try:
            dest = os.path.join(docs_dir, safe_name)
            with open(dest, 'wb+') as out:
                for chunk in f.chunks():
                    out.write(chunk)
            saved.append(safe_name)
        except Exception as e:
            logger.error("Falha ao salvar documento RAG %s: %s", safe_name, e)
            skipped.append({"name": safe_name, "reason": "Falha ao salvar no servidor."})

    rebuilt = False
    if saved:
        try:
            rebuilt = RagService().clear_and_rebuild_storage()
        except Exception as e:
            logger.error("Falha ao reconstruir índice RAG após upload: %s", e)

    return Response({
        "saved": saved,
        "skipped": skipped,
        "rebuilt": rebuilt,
        "documents": _list_rag_documents(),
    })


@api_view(['DELETE'])
def rag_documents_delete(request, filename):
    """
    Remove um arquivo da pasta documents/ e reconstrói o índice RAG.
    """
    from .services.rag_service import RagService

    safe_name = _sanitize_doc_filename(filename)
    docs_dir = _rag_documents_dir()
    target = os.path.abspath(os.path.join(docs_dir, safe_name))
    if not safe_name or not target.startswith(os.path.abspath(docs_dir) + os.sep):
        return Response({"error": "Nome de arquivo inválido."},
                        status=status.HTTP_400_BAD_REQUEST)
    if not os.path.isfile(target):
        return Response({"error": "Arquivo não encontrado."},
                        status=status.HTTP_404_NOT_FOUND)
    try:
        os.unlink(target)
    except Exception as e:
        logger.error("Falha ao remover documento RAG %s: %s", safe_name, e)
        return Response({"error": "Falha ao remover o arquivo."},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    rebuilt = False
    try:
        rebuilt = RagService().clear_and_rebuild_storage()
    except Exception as e:
        logger.error("Falha ao reconstruir índice RAG após remoção: %s", e)

    return Response({"status": "deleted", "name": safe_name,
                     "rebuilt": rebuilt, "documents": _list_rag_documents()})


@api_view(['POST'])
def unload_model(request):
    """
    Endpoint para descarregar o modelo atual da memória.
    """
    success = lmm_service.unload_model()
    if success:
        return Response({"status": "Model unloaded successfully and memory freed!"}, status=status.HTTP_200_OK)
    return Response({"error": "Failed to unload model"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ── Histórico de conversas (persistido no servidor) ──
def _conv_to_list(c):
    return {
        "id": c.id,
        "title": c.title,
        "created_at": c.created_at,
        "updated_at": c.updated_at,
        "message_count": c.messages.count(),
    }


def _conv_to_detail(c):
    return {
        "id": c.id,
        "title": c.title,
        "created_at": c.created_at,
        "updated_at": c.updated_at,
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "image_base64": m.image_base64 or "",
                "created_at": m.created_at,
            }
            for m in c.messages.all()
        ],
    }


@api_view(['GET', 'POST'])
def conversations_list_create(request):
    from .models import Conversation
    from .serializers import ConversationCreateSerializer, ConversationListSerializer

    if request.method == 'GET':
        convs = Conversation.objects.prefetch_related('messages').all()
        data = [_conv_to_list(c) for c in convs]
        return Response(ConversationListSerializer(data, many=True).data)

    serializer = ConversationCreateSerializer(data=request.data or {})
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    title = (serializer.validated_data.get('title') or '').strip() or 'Nova conversa'
    conv = Conversation.objects.create(title=title[:255])
    return Response(_conv_to_detail(conv), status=status.HTTP_201_CREATED)


@api_view(['GET', 'PATCH', 'DELETE'])
def conversation_detail(request, conv_id):
    from .models import Conversation
    from .serializers import ConversationUpdateSerializer

    try:
        conv = Conversation.objects.prefetch_related('messages').get(id=conv_id)
    except Conversation.DoesNotExist:
        return Response({"error": "Conversa não encontrada"}, status=status.HTTP_404_NOT_FOUND)

    if request.method == 'GET':
        return Response(_conv_to_detail(conv))

    if request.method == 'PATCH':
        serializer = ConversationUpdateSerializer(data=request.data or {})
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        conv.title = serializer.validated_data['title'].strip()[:255] or conv.title
        conv.save(update_fields=['title', 'updated_at'])
        return Response(_conv_to_detail(conv))

    conv.delete()
    return Response({"status": "deleted"})


@api_view(['POST'])
def conversation_append_message(request, conv_id):
    from .models import Conversation, ChatMessage
    from .serializers import ChatMessageSerializer

    try:
        conv = Conversation.objects.get(id=conv_id)
    except Conversation.DoesNotExist:
        return Response({"error": "Conversa não encontrada"}, status=status.HTTP_404_NOT_FOUND)

    serializer = ChatMessageSerializer(data=request.data or {})
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    msg = ChatMessage.objects.create(
        conversation=conv,
        role=serializer.validated_data['role'],
        content=serializer.validated_data['content'],
        image_base64=serializer.validated_data.get('image_base64') or '',
    )
    # atualiza título automático na primeira mensagem de usuário
    data = serializer.validated_data
    if conv.messages.count() <= 1 and data['role'] == 'user' and conv.title == 'Nova conversa':
        t = ' '.join((data['content'] or '').split())[:38]
        conv.title = (t + '…') if len((data['content'] or '')) > 38 else (t or conv.title)
        conv.save(update_fields=['title', 'updated_at'])
    else:
        conv.save(update_fields=['updated_at'])

    return Response(ChatMessageSerializer({
        "id": msg.id,
        "role": msg.role,
        "content": msg.content,
        "image_base64": msg.image_base64,
        "created_at": msg.created_at,
    }).data, status=status.HTTP_201_CREATED)


@api_view(['POST'])
def conversations_clear(request):
    from .models import Conversation
    deleted, _ = Conversation.objects.all().delete()
    return Response({"status": "cleared", "deleted_objects": deleted})
