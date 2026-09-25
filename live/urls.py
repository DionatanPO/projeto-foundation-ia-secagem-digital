from django.urls import path
from .views import live_health, live_voices, live_speak, live_split, live_transcribe

urlpatterns = [
    path('health/', live_health, name='live-health'),
    path('voices/', live_voices, name='live-voices'),
    path('speak/', live_speak, name='live-speak'),
    path('split/', live_split, name='live-split'),
    path('transcribe/', live_transcribe, name='live-transcribe'),
]
