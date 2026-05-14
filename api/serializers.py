from rest_framework import serializers

class ChatRequestSerializer(serializers.Serializer):
    prompt = serializers.CharField(required=True, help_text="O texto para enviar ao modelo.")
    temperature = serializers.FloatField(required=False, default=0.7, help_text="Criatividade/Aleatoriedade (0.0 a 1.0).")
    image_base64 = serializers.CharField(required=False, allow_null=True, allow_blank=True, help_text="Imagem em base64 para modelos visuais.")
    history = serializers.ListField(child=serializers.DictField(), required=False, default=[], help_text="Histórico da conversa.")
    system_prompt = serializers.CharField(required=False, allow_null=True, allow_blank=True, help_text="Diretrizes personalizadas para a IA.")

class ModelSwitchSerializer(serializers.Serializer):
    model_name = serializers.CharField()
    use_gpu = serializers.BooleanField(required=False, default=False)

class ChatResponseSerializer(serializers.Serializer):
    response = serializers.CharField()
