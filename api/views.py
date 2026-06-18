from rest_framework.decorators import api_view, permission_classes, authentication_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from django.http import StreamingHttpResponse, JsonResponse
import logging
import json
import psutil
import os
from .serializers import ChatRequestSerializer, ChatResponseSerializer, ModelSwitchSerializer

logger = logging.getLogger(__name__)
from .services.lmm_service import LMMService
from .services.remote_llm_service import RemoteLLMService

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
    Retorna o consumo de memória RAM do processo atual (Django + LMM) e do sistema.
    """
    process = psutil.Process(os.getpid())
    process_memory_mb = process.memory_info().rss / (1024 * 1024)
    system_memory = psutil.virtual_memory()
    total_memory_mb = system_memory.total / (1024 * 1024)
    used_memory_mb = system_memory.used / (1024 * 1024)
    memory_percent = system_memory.percent

    return Response({
        "process_ram_mb": round(process_memory_mb, 2),
        "system_total_mb": round(total_memory_mb, 2),
        "system_used_mb": round(used_memory_mb, 2),
        "system_percent": memory_percent
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
            use_rag=use_rag
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
                    use_rag=use_rag
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
    return Response({
        "mode": "remote" if enabled else "local",
        "remote_enabled": enabled,
        "remote_api_url": cfg.get("api_url", ""),
        "local_model_loaded": lmm_service.model is not None,
        "local_model": lmm_service.get_current_model()
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
    return Response({
        "models": models,
        "current_model": current_model
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
            return Response({"status": "Model switched", "model": model_name})
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

@api_view(['POST'])
def unload_model(request):
    """
    Endpoint para descarregar o modelo atual da memória.
    """
    success = lmm_service.unload_model()
    if success:
        return Response({"status": "Model unloaded successfully and memory freed!"}, status=status.HTTP_200_OK)
    return Response({"error": "Failed to unload model"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
