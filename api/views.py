from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import StreamingHttpResponse
import psutil
import os
from .serializers import ChatRequestSerializer, ChatResponseSerializer, ModelSwitchSerializer
from .services.lmm_service import LMMService

# Instancia o serviço (O modelo será carregado na primeira vez que esta view for importada)
lmm_service = LMMService()

@api_view(['GET'])
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
    """
    serializer = ChatRequestSerializer(data=request.data)

    if serializer.is_valid():
        prompt = serializer.validated_data['prompt']
        temperature = max(0.0, min(1.0, serializer.validated_data['temperature']))
        image_base64 = serializer.validated_data.get('image_base64', None)
        system_prompt = serializer.validated_data.get('system_prompt', None)
        history = serializer.validated_data.get('history', [])
        use_rag = serializer.validated_data.get('use_rag', True)

        # Usamos generate_response que já trata o JSON interno do llama.cpp
        # Nota: Ajustaremos o LMMService se necessário para separar melhor.
        answer_text = lmm_service.generate_response(
            prompt=prompt,
            temperature=temperature,
            image_base64=image_base64,
            system_prompt=system_prompt,
            history=history,
            use_rag=use_rag
        )

        # Resposta formatada para consumo simples
        return Response({
            'resposta': answer_text,
            'thinking': "" # Opcional: extrair do modelo se necessário
        }, status=status.HTTP_200_OK)

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
def chat_stream(request):
    """
    Endpoint de streaming para resposta em tempo real.
    """
    serializer = ChatRequestSerializer(data=request.data)
    if serializer.is_valid():
        prompt = serializer.validated_data['prompt']
        temperature = serializer.validated_data.get('temperature', 0.2)
        image_base64 = serializer.validated_data.get('image_base64', None)
        system_prompt = serializer.validated_data.get('system_prompt', None)
        history = serializer.validated_data.get('history', [])
        use_rag = serializer.validated_data.get('use_rag', True)

        def stream_generator():
            try:
                for chunk in lmm_service.generate_stream(
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
