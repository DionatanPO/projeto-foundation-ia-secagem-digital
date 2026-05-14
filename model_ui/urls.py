from django.urls import path
from .views import chat_interface

urlpatterns = [
    path('', chat_interface, name='chat-interface'),
]
