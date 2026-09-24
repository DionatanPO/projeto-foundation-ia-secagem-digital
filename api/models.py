from django.db import models


class RemoteConfig(models.Model):
    enabled = models.BooleanField(default=False)
    api_url = models.CharField(max_length=500, blank=True, default='')
    model = models.CharField(max_length=200, blank=True, default='')
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Configuração Remota"
        verbose_name_plural = "Configurações Remotas"

    def __str__(self):
        return f"Remote({'ON' if self.enabled else 'OFF'}) {self.api_url or 'sem URL'}"


class Conversation(models.Model):
    title = models.CharField(max_length=255, default='Nova conversa')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']
        verbose_name = "Conversa"
        verbose_name_plural = "Conversas"

    def __str__(self):
        return f"{self.id} - {self.title}"


class ChatMessage(models.Model):
    ROLE_CHOICES = (('user', 'user'), ('assistant', 'assistant'))

    conversation = models.ForeignKey(
        Conversation, related_name='messages', on_delete=models.CASCADE
    )
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    image_base64 = models.TextField(blank=True, default='')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']
        verbose_name = "Mensagem"
        verbose_name_plural = "Mensagens"

    def __str__(self):
        return f"{self.conversation_id}/{self.role} #{self.id}"
