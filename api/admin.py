from django.contrib import admin
from .models import RemoteConfig, Conversation, ChatMessage


@admin.register(RemoteConfig)
class RemoteConfigAdmin(admin.ModelAdmin):
    list_display = ('enabled', 'api_url', 'model', 'updated_at')


class ChatMessageInline(admin.TabularInline):
    model = ChatMessage
    extra = 0
    readonly_fields = ('role', 'created_at')
    fields = ('role', 'content', 'created_at')


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'title', 'created_at', 'updated_at')
    list_filter = ('user',)
    search_fields = ('title', 'user__email', 'user__username')
    inlines = [ChatMessageInline]


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ('id', 'conversation', 'role', 'created_at')
    list_filter = ('role',)
