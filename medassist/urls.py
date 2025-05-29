from django.urls import path
from . import views

urlpatterns = [
    path('chat/', views.chat_view, name='chat_page'), # Your main chat page
    path('api/chat/', views.chat_api_endpoint, name='chat_api'), # API endpoint for chat messages/uploads
    path('api/initialize-csrf/', views.initialize_csrf, name='initialize_csrf'), # CSRF initialization endpoint for Next.js
    path('api/login/', views.login_api, name='login_api'), # CSRF-protected login endpoint
    path('api/signup/', views.signup_api, name='signup_api'), # CSRF-protected signup endpoint
    path('api/logout/', views.logout_api, name='logout_api'), # CSRF-protected logout endpoint
    path('api/user-status/', views.get_user_status, name='user_status'), # Check user authentication status
]