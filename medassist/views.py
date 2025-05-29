# DoctorAssistant/medassist/views.py

from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie, csrf_protect
from django.views.decorators.http import require_GET, require_POST
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from .models import MedicalDocument, MedicalAssistantInteraction
from .ai_service import MedicalAIService
import json

@ensure_csrf_cookie
@require_GET
def initialize_csrf(request):
    """
    Dedicated endpoint to ensure CSRF cookie is set for Next.js frontend.
    Call this endpoint when your Next.js app initializes to guarantee 
    the csrftoken cookie is available for subsequent POST requests.
    """
    return JsonResponse({
        "status": "csrf_cookie_set",
        "message": "CSRF token has been initialized"
    })

@ensure_csrf_cookie
def home(request):
    """
    Home view with CSRF cookie ensured for frontend integration
    """
    return HttpResponse("Welcome to DoctorAssistant Home Page!")

# Authentication API endpoints
@csrf_protect
@require_POST
def login_api(request):
    """
    CSRF-protected login API endpoint for Next.js frontend.
    Expects application/x-www-form-urlencoded data with:
    - username (or email)
    - password
    - csrfmiddlewaretoken (form field) or X-CSRFToken (header)
    """
    try:
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        if not username or not password:
            return JsonResponse({
                "status": "error",
                "message": "Username and password are required"
            }, status=400)
        
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            return JsonResponse({
                "status": "success",
                "message": "Login successful",
                "user": {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email
                }
            })
        else:
            return JsonResponse({
                "status": "error",
                "message": "Invalid username or password"
            }, status=401)
            
    except Exception as e:
        return JsonResponse({
            "status": "error",
            "message": f"An error occurred during login: {str(e)}"
        }, status=500)

@csrf_protect
@require_POST
def signup_api(request):
    """
    CSRF-protected signup API endpoint for Next.js frontend.
    Expects application/x-www-form-urlencoded data with:
    - username
    - email
    - password
    - csrfmiddlewaretoken (form field) or X-CSRFToken (header)
    """
    try:
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        
        if not all([username, email, password]):
            return JsonResponse({
                "status": "error",
                "message": "Username, email, and password are required"
            }, status=400)
        
        # Check if user already exists
        if User.objects.filter(username=username).exists():
            return JsonResponse({
                "status": "error",
                "message": "Username already exists"
            }, status=400)
            
        if User.objects.filter(email=email).exists():
            return JsonResponse({
                "status": "error",
                "message": "Email already exists"
            }, status=400)
        
        # Create new user
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password
        )
        
        # Automatically log in the new user
        login(request, user)
        
        return JsonResponse({
            "status": "success",
            "message": "Account created successfully",
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email
            }
        })
        
    except Exception as e:
        return JsonResponse({
            "status": "error",
            "message": f"An error occurred during signup: {str(e)}"
        }, status=500)

# This will be your main chat page
@login_required # Ensure user is logged in
@ensure_csrf_cookie
def chat_view(request):
    return render(request, 'medassist/chat.html') # We'll create this template next

# This will be your API endpoint for chat messages and file uploads with proper CSRF protection
@csrf_protect
@login_required
@require_POST
def chat_api_endpoint(request):
    """
    CSRF-protected API endpoint for handling chat requests from Next.js frontend.
    Now includes AI analysis using OpenAI.
    """
    user = request.user
    ai_service = MedicalAIService()
    
    try:
        if 'file' in request.FILES:
            uploaded_file = request.FILES['file']
            # Save the file to MedicalDocument model
            medical_doc = MedicalDocument(
                user=user,
                document_type='PDF' if uploaded_file.name.endswith('.pdf') else 'IMAGE',
                file=uploaded_file
            )
            medical_doc.save()
            
            # TODO: Add OCR extraction for files here
            # For now, we'll use a placeholder text
            file_content = f"File uploaded: {uploaded_file.name}"
            
            # Get AI analysis for the document
            ai_result = ai_service.analyze_medical_document(
                document_text=file_content,
                document_type=medical_doc.document_type
            )
            
            # Save the interaction
            interaction = MedicalAssistantInteraction(
                user=user,
                input_query=f"File upload: {uploaded_file.name}",
                reasoning_output=ai_result.get('ai_response', 'No analysis available'),
                final_analysis=ai_result.get('ai_response', 'No analysis available')
            )
            interaction.save()
            
            response_data = {
                "status": "success",
                "document_id": medical_doc.id,
                "input_type": "file_upload",
                "message": ai_result.get('ai_response', f"File '{uploaded_file.name}' uploaded successfully.")
            }
            
        elif 'text_input' in request.POST:
            text_input = request.POST['text_input']
            
            # Save the text to MedicalDocument model
            medical_doc = MedicalDocument(
                user=user,
                document_type='TEXT',
                raw_text=text_input
            )
            medical_doc.save()
            
            # Get user's previous medical history for context
            previous_interactions = MedicalAssistantInteraction.objects.filter(
                user=user
            ).order_by('-timestamp')[:3]
            
            user_history = ""
            if previous_interactions:
                history_parts = []
                for interaction in previous_interactions:
                    history_parts.append(f"Previous query: {interaction.input_query[:100]}")
                user_history = " | ".join(history_parts)
            
            # Get AI analysis for the text
            ai_result = ai_service.analyze_medical_text(
                text_input=text_input,
                user_history=user_history if user_history else None
            )
            
            # Save the interaction
            interaction = MedicalAssistantInteraction(
                user=user,
                input_query=text_input,
                reasoning_output=ai_result.get('ai_response', 'No analysis available'),
                final_analysis=ai_result.get('ai_response', 'No analysis available')
            )
            interaction.save()
            
            response_data = {
                "status": "success",
                "document_id": medical_doc.id,
                "input_type": "text_input",
                "message": ai_result.get('ai_response', f"I've received your message: '{text_input[:50]}...' Let me analyze this for you.")
            }
            
        else:
            response_data = {
                "status": "error",
                "message": "No file or text_input found in the request."
            }

    except Exception as e:
        response_data = {
            "status": "error",
            "message": f"An error occurred while processing your request: {str(e)}"
        }
        print(f"Error in chat_api_endpoint: {e}")

    return JsonResponse(response_data)

@csrf_protect
@login_required
@require_POST  
def logout_api(request):
    """
    API endpoint for logout that properly handles CSRF tokens
    """
    logout(request)
    return JsonResponse({"status": "success", "message": "Logged out successfully"})

@ensure_csrf_cookie
@require_GET
def get_user_status(request):
    """
    API endpoint to check user authentication status and ensure CSRF cookie is set
    """
    if request.user.is_authenticated:
        return JsonResponse({
            "authenticated": True,
            "username": request.user.username,
            "user_id": request.user.id
        })
    else:
        return JsonResponse({
            "authenticated": False
        })