from django.contrib import admin
from .models import RemoteConfig


@admin.register(RemoteConfig)
class RemoteConfigAdmin(admin.ModelAdmin):
    list_display = ('enabled', 'api_url', 'model', 'updated_at')
