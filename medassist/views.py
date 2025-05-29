# DoctorAssistant/medassist/views.py

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from .models import MedicalDocument # Import your model
from django.http import HttpResponse

def home(request):
    return HttpResponse("Welcome to DoctorAssistant Home Page!")
# This will be your main chat page
@login_required # Ensure user is logged in
def chat_view(request):
    return render(request, 'medassist/chat.html') # We'll create this template next

# This will be your API endpoint for chat messages and file uploads
@csrf_exempt # For simplicity in development, consider proper CSRF token handling in production
@login_required
def chat_api_endpoint(request):
    if request.method == 'POST':
        user = request.user
        response_data = {"status": "success", "message": "Received input."}

        if 'file' in request.FILES:
            uploaded_file = request.FILES['file']
            # Save the file to MedicalDocument model
            medical_doc = MedicalDocument(
                user=user,
                document_type='PDF' if uploaded_file.name.endswith('.pdf') else 'IMAGE',
                file=uploaded_file
            )
            medical_doc.save()
            response_data["document_id"] = medical_doc.id
            response_data["input_type"] = "file_upload"
            response_data["message"] = "File uploaded successfully."
            # Here, you'd trigger your LangGraph run with the new document_id
            # For now, just logging
            print(f"File saved: {medical_doc.file.path}")

        elif 'text_input' in request.POST:
            text_input = request.POST['text_input']
            # Save the text to MedicalDocument model
            medical_doc = MedicalDocument(
                user=user,
                document_type='TEXT',
                raw_text=text_input
            )
            medical_doc.save()
            response_data["document_id"] = medical_doc.id
            response_data["input_type"] = "text_input"
            response_data["message"] = "Text input received successfully."
            # Here, you'd trigger your LangGraph run with the new document_id
            # For now, just logging
            print(f"Text saved: {medical_doc.raw_text}")
        else:
            response_data["status"] = "error"
            response_data["message"] = "No file or text_input found."

        return JsonResponse(response_data)
    return JsonResponse({"status": "error", "message": "Only POST requests allowed."})