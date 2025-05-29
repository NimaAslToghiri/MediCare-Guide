from django.contrib import admin 
from django.urls import path, include, re_path
from django.conf import settings
from django.conf.urls.static import static  # ✅ You need this import
from django.views.static import serve
from medassist.views import home  # Or your custom home view

urlpatterns = [
    path('admin/', admin.site.urls),  # Admin panel
    path('accounts/', include('django.contrib.auth.urls')),  # Add authentication URLs
    path('medassist/', include('medassist.urls')),  # Include URLs from the medassist app
    re_path(r'^media/(?P<path>.*)$', serve, {'document_root': settings.MEDIA_ROOT}),  # Serve media files in development

    path('', home),  # Home page for the root URL
    # Alternatively:
    # path('', lambda request: redirect('medassist/', permanent=False)),  # Redirect root URL to /medassist/
]

# ONLY add this for development purposes to serve media files
# In production, you would use a dedicated web server (e.g., Nginx) for media files
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
