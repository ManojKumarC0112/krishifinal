# krishi_sakhi/krishi_sakhi/urls.py (Project-level)

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # Django Admin Panel
    path('admin/', admin.site.urls),
    
    # Route all main application URLs to the 'main' app's urls.py
    path('', include('main.urls')), 
]

# Serving media files during development (Crucial for video/image uploads)
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)