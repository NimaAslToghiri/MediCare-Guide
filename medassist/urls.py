from django.urls import path
from . import views

urlpatterns = [
    # Chat views
    path('chat/', views.chat_view, name='chat_page'),
    path('chat', views.chat_view, name='chat_page_no_slash'),
    
    # API endpoints - handle both with and without trailing slashes
    path('api/chat/', views.chat_api_endpoint, name='chat_api'),
    path('api/chat', views.chat_api_endpoint, name='chat_api_no_slash'),
    path('api/login/', views.login_api, name='login_api'),
    path('api/login', views.login_api, name='login_api_no_slash'),
    path('api/signup/', views.signup_api, name='signup_api'),
    path('api/signup', views.signup_api, name='signup_api_no_slash'),
    path('api/logout/', views.logout_api, name='logout_api'),
    path('api/logout', views.logout_api, name='logout_api_no_slash'),
    path('api/user-status/', views.get_user_status, name='user_status'),
    path('api/user-status', views.get_user_status, name='user_status_no_slash'),
    
    # CSRF initialization (also available at app level)
    path('api/initialize-csrf/', views.initialize_csrf, name='initialize_csrf_medassist'),
    path('api/initialize-csrf', views.initialize_csrf, name='initialize_csrf_medassist_no_slash'),
]