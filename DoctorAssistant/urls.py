# DoctorAssistant/DoctorAssistant/urls.py

from django.contrib import admin
from django.urls import path, include # Import include
from django.conf import settings # Import settings
from django.conf.urls.static import static # Import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('medassist/', include('medassist.urls')), # Include your app's URLs
]

# ONLY add this for development purposes to serve media files
# In production, you would use a dedicated web server (e.g., Nginx) for media files
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)