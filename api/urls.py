from django.urls import path
from .views import health_check, chat_inference, system_status, chat_stream, list_models, switch_model

urlpatterns = [
    path('health/', health_check, name='health-check'),
    path('chat/', chat_inference, name='chat-inference'),
    path('chat-stream/', chat_stream, name='chat-stream'),
    path('status/', system_status, name='system-status'),
    path('models/', list_models, name='list-models'),
    path('switch-model/', switch_model, name='switch-model'),
]
