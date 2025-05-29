from django.urls import path
from . import views

urlpatterns = [
    path('chat/', views.chat_view, name='chat_page'), # Your main chat page
    path('api/chat/', views.chat_api_endpoint, name='chat_api'), # API endpoint for chat messages/uploads
    # Add other API endpoints as needed, e.g., for user history if separate
]