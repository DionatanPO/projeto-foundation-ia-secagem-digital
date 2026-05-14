from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import StreamingHttpResponse
import psutil
import os
from .serializers import ChatRequestSerializer, ChatResponseSerializer
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
    Endpoint para enviar um prompt ao modelo LMM e obter a resposta.
    """
    serializer = ChatRequestSerializer(data=request.data)

    if serializer.is_valid():
        prompt = serializer.validated_data['prompt']
        max_tokens = serializer.validated_data.get('max_tokens', 512)
        temperature = serializer.validated_data.get('temperature', 0.7)
        image_base64 = serializer.validated_data.get('image_base64', None)
        use_think = serializer.validated_data.get('use_think', False)
        system_prompt = serializer.validated_data.get('system_prompt', None)
        history = serializer.validated_data.get('history', [])

        answer = lmm_service.generate_response(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            image_base64=image_base64,
            use_think=use_think,
            system_prompt=system_prompt,
            history=history
        )

        return Response({'response': answer}, status=status.HTTP_200_OK)

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def chat_stream(request):
    """
    Endpoint de streaming para resposta em tempo real.
    """
    serializer = ChatRequestSerializer(data=request.data)
    if serializer.is_valid():
        prompt = serializer.validated_data['prompt']
        max_tokens = serializer.validated_data.get('max_tokens', 512)
        temperature = serializer.validated_data.get('temperature', 0.7)
        image_base64 = serializer.validated_data.get('image_base64', None)
        use_think = serializer.validated_data.get('use_think', False)
        system_prompt = serializer.validated_data.get('system_prompt', None)
        history = serializer.validated_data.get('history', [])

        def stream_generator():
            for chunk in lmm_service.generate_stream(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                image_base64=image_base64,
                use_think=use_think,
                system_prompt=system_prompt,
                history=history
            ):
                yield chunk

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
    model_name = request.data.get('model_name')
    if not model_name:
        return Response({"error": "Nome do modelo não fornecido"}, status=status.HTTP_400_BAD_REQUEST)

    success = lmm_service.switch_model(model_name)
    if success:
        return Response({"message": f"Modelo {model_name} carregado com sucesso"}, status=status.HTTP_200_OK)
    else:
        return Response({"error": f"Falha ao carregar o modelo {model_name}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
