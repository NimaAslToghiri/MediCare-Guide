import os
import openai
from dotenv import load_dotenv
from .models import ChatMessage, MedicalDocument
from .forms import MedicalDocumentForm
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.views.decorators.csrf import ensure_csrf_cookie, csrf_protect
from django.views.decorators.http import require_http_methods
import mimetypes
from .agents import intro_agent_with_llm, input_validator_agent, reasoning_agent, revisor_agent
import mimetypes

# Load environment variables
load_dotenv()

def get_user_history_summary(user):
    """Get a summary of user's medical document history"""
    docs = MedicalDocument.objects.filter(user=user).order_by('-uploaded_at')
    if not docs.exists():
        return "User has no prior medical documents."

    summaries = []
    for doc in docs[:3]:  # Limit to last 3 documents
        summaries.append(f"- {doc.file.name} uploaded on {doc.uploaded_at.date()}")
    return "\n".join(summaries)

@login_required
@ensure_csrf_cookie
@require_http_methods(["GET", "POST"])
def chat_page(request):
    """
    Chat page view with proper CSRF token handling for frontend integration
    """
    form = MedicalDocumentForm()
    intro_message = intro_agent_with_llm(request.user)
    chat_log = ChatMessage.objects.filter(user=request.user).order_by('timestamp')

    if request.method == 'POST':
        # CSRF protection is automatically handled by the middleware
        # since we're using the @csrf_protect decorator (implied by middleware)
        
        form = MedicalDocumentForm(request.POST, request.FILES)
        user_message = request.POST.get("user_message", "").strip()

        document_saved = False
        message_saved = False
        is_valid = True  # Initialize is_valid

        # 🧾 Save file if any
        if request.FILES.get("file"):
            if form.is_valid():
                document = form.save(commit=False)
                document.user = request.user

                # MIME detection
                mime_type, _ = mimetypes.guess_type(document.file.name)
                if mime_type:
                    if 'pdf' in mime_type:
                        document.file_type = 'pdf'
                    elif 'image' in mime_type:
                        document.file_type = 'image'
                    elif 'text' in mime_type:
                        document.file_type = 'text'
                    else:
                        document.file_type = 'text'  # fallback

                document.save()
                document_saved = True

                # Log in chat
                ChatMessage.objects.create(user=request.user, message=f"📎 Uploaded file: {document.file.name}", sender='user')

        # 🧠 Save message if any
        if user_message:
            is_valid = input_validator_agent(user_message)
            ChatMessage.objects.create(user=request.user, message=user_message, sender='user')
            
            if not is_valid:
                ChatMessage.objects.create(user=request.user, message="⚠️ Please provide a medically relevant message or upload a report.", sender='agent')
            message_saved = True

        # 🤖 Agent Response if valid message
        if message_saved and is_valid:
            try:
                reasoning = reasoning_agent(user_message, get_user_history_summary(request.user))
                final_reply = revisor_agent(reasoning, user_message, get_user_history_summary(request.user))
                ChatMessage.objects.create(user=request.user, message=final_reply, sender='agent')
            except Exception as e:
                ChatMessage.objects.create(user=request.user, message="Sorry, I encountered an error processing your request. Please try again.", sender='agent')
                print(f"Agent error: {e}")

        return redirect("chat_page")

    return render(request, "assistant/chat.html", {
        "form": form,
        "message": intro_message,
        "chat_history": chat_log,  # Changed from chat_log to match template
    })