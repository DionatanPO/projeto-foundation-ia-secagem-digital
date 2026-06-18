from django.urls import path
from .views import health_check, chat_inference, system_status, chat_stream, list_models, switch_model, clear_rag_storage, unload_model, remote_config_save, remote_config_load, remote_config_test, service_mode

urlpatterns = [
    path('health/', health_check, name='health-check'),
    path('chat/', chat_inference, name='chat-inference'),
    path('chat-stream/', chat_stream, name='chat-stream'),
    path('status/', system_status, name='system-status'),
    path('models/', list_models, name='list-models'),
    path('switch-model/', switch_model, name='switch-model'),
    path('clear-rag/', clear_rag_storage, name='clear-rag'),
    path('unload-model/', unload_model, name='unload-model'),
    path('remote-config/save/', remote_config_save, name='remote-config-save'),
    path('remote-config/load/', remote_config_load, name='remote-config-load'),
    path('remote-config/test/', remote_config_test, name='remote-config-test'),
    path('service-mode/', service_mode, name='service-mode'),
]
