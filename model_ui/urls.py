from django.urls import path
from .views import chat_interface, landing_view, login_view, logout_view

urlpatterns = [
    path('', landing_view, name='inicio'),
    path('solumai/', landing_view, name='solumai-landing'),
    path('chat/', chat_interface, name='chat-interface'),
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
]
