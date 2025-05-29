import os
import openai
from dotenv import load_dotenv
from .models import ChatMessage, MedicalDocument
from .forms import MedicalDocumentForm
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
import mimetypes
from .agents import intro_agent_with_llm, input_validator_agent, reasoning_agent, revisor_agent
import mimetypes


@login_required
def chat_page(request):
    form = MedicalDocumentForm()
    intro_message = intro_agent_with_llm(request.user)
    chat_log = ChatMessage.objects.filter(user=request.user).order_by('timestamp')

    if request.method == 'POST':
        form = MedicalDocumentForm(request.POST, request.FILES)
        user_message = request.POST.get("user_message", "").strip()

        document_saved = False
        message_saved = False

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
            if is_valid:
                ChatMessage.objects.create(user=request.user, message=user_message, sender='user')
            else:
                ChatMessage.objects.create(user=request.user, message=user_message, sender='user')
                ChatMessage.objects.create(user=request.user, message="⚠️ Please provide a medically relevant message or upload a report.", sender='agent')
            message_saved = True

        # 🤖 Agent Response if valid message
        if message_saved and is_valid:
            context = "\n".join([
                f"{msg.sender.capitalize()}: {msg.message}" for msg in chat_log
            ]) + f"\nUser: {user_message}"

            reasoning = reasoning_agent(user_message, get_user_history_summary(request.user))
            final_reply = revisor_agent(reasoning, user_message, get_user_history_summary(request.user))
            ChatMessage.objects.create(user=request.user, message=final_reply, sender='agent')

        return redirect("chat_page")

    return render(request, "assistant/chat.html", {
        "form": form,
        "message": intro_message,
        "chat_log": chat_log,
    })