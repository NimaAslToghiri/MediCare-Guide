from django.contrib import admin 
from django.urls import path, include, re_path
from django.conf import settings
from django.conf.urls.static import static  # ✅ You need this import
from django.views.static import serve
from django.contrib.auth import views as auth_views
from medassist.views import home, initialize_csrf  # Import the CSRF initialization view

urlpatterns = [
    path('admin/', admin.site.urls),  # Admin panel
    path('medassist/', include('medassist.urls')),  # Include URLs from the medassist app
    path('assistant/', include('assistant.urls')),  # Include URLs from the assistant app
    
    # CSRF initialization endpoint for Next.js frontend (direct route)
    path('api/initialize-csrf/', initialize_csrf, name='initialize_csrf'),
    
    # Authentication URLs
    path('login/', auth_views.LoginView.as_view(), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    
    re_path(r'^media/(?P<path>.*)$', serve, {'document_root': settings.MEDIA_ROOT}),  # Serve media files in development

    path('', home, name='home'),  # Home page for the root URL
]

# ONLY add this for development purposes to serve media files
# In production, you would use a dedicated web server (e.g., Nginx) for media files
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
