from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('api.urls')),
    path('api/live/', include('live.urls')),
    path('', include('model_ui.urls')),
]
