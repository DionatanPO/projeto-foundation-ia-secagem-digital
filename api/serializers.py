from rest_framework import serializers

class ChatRequestSerializer(serializers.Serializer):
    prompt = serializers.CharField(required=True, help_text="O texto para enviar ao modelo.")
    temperature = serializers.FloatField(required=False, default=0.1, help_text="Criatividade/Aleatoriedade (0.0 a 1.0).")
    image_base64 = serializers.CharField(required=False, allow_null=True, allow_blank=True, help_text="Imagem em base64 para modelos visuais.")
    history = serializers.ListField(child=serializers.DictField(), required=False, default=[], help_text="Histórico da conversa.")
    system_prompt = serializers.CharField(required=False, allow_null=True, allow_blank=True, help_text="Diretrizes personalizadas para a IA.")
    use_rag = serializers.BooleanField(required=False, default=False, help_text="Se deve utilizar RAG (banco vetorial) para enriquecer o contexto.")
    use_remote = serializers.BooleanField(required=False, default=None, allow_null=True, help_text="Se deve usar modelo remoto. Se não enviado, usa a configuração salva no servidor.")
    remote_config = serializers.DictField(required=False, default={}, help_text="Configuração do modelo remoto (api_url, api_key, model). Opcional se já configurado.")
    attachments = serializers.ListField(child=serializers.DictField(), required=False, default=[], help_text="Arquivos anexados: [{filename, mime, content_base64}]. Texto extraído e injetado como contexto.")
    max_tokens = serializers.IntegerField(required=False, allow_null=True, default=None, min_value=32, max_value=4096, help_text="Teto de tokens da resposta (ex: modo Live usa ~256). Omitido = sem limite.")

class ModelSwitchSerializer(serializers.Serializer):
    model_name = serializers.CharField()
    use_gpu = serializers.BooleanField(required=False, default=False)

class ChatResponseSerializer(serializers.Serializer):
    response = serializers.CharField()


class ChatMessageSerializer(serializers.Serializer):
    id = serializers.IntegerField(read_only=True)
    role = serializers.ChoiceField(choices=['user', 'assistant'])
    content = serializers.CharField()
    image_base64 = serializers.CharField(required=False, allow_blank=True, default='')
    created_at = serializers.DateTimeField(read_only=True)


class ConversationListSerializer(serializers.Serializer):
    id = serializers.IntegerField(read_only=True)
    title = serializers.CharField()
    created_at = serializers.DateTimeField(read_only=True)
    updated_at = serializers.DateTimeField(read_only=True)
    message_count = serializers.IntegerField(read_only=True)


class ConversationDetailSerializer(serializers.Serializer):
    id = serializers.IntegerField(read_only=True)
    title = serializers.CharField()
    created_at = serializers.DateTimeField(read_only=True)
    updated_at = serializers.DateTimeField(read_only=True)
    messages = ChatMessageSerializer(many=True, read_only=True)


class ConversationCreateSerializer(serializers.Serializer):
    title = serializers.CharField(required=False, allow_blank=True, default='Nova conversa')


class ConversationUpdateSerializer(serializers.Serializer):
    title = serializers.CharField(required=True, allow_blank=False)
