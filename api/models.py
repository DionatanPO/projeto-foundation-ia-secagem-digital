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
