from django.urls import path
from .views import chat_interface, login_view

urlpatterns = [
    path('', chat_interface, name='chat-interface'),
    path('login/', login_view, name='login'),
]
