# DoctorAssistant/medassist/models.py

from django.db import models
from django.contrib.auth.models import User # Using Django's built-in User model

class MedicalDocument(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='medical_documents')
    document_type_choices = [
        ('PDF', 'PDF Document'),
        ('IMAGE', 'Image File'),
        ('TEXT', 'Plain Text'), # For direct text input or OCR results stored here
    ]
    document_type = models.CharField(
        max_length=10,
        choices=document_type_choices,
        default='TEXT',
        help_text="Type of the document (e.g., PDF, Image, Plain Text)."
    )
    file = models.FileField(
        upload_to='medical_documents/',
        blank=True,
        null=True,
        help_text="Uploaded PDF or image file."
    )
    raw_text = models.TextField( # This is where OCR text will be saved
        blank=True,
        null=True,
        help_text="Raw text content from user input or OCR extraction."
    )
    upload_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username}'s {self.document_type} on {self.upload_date.strftime('%Y-%m-%d')}"


class ExtractedMedicalData(models.Model):
    """
    Stores structured medical data extracted from documents or user input
    by the OCR & Initial Extraction Agent.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='extracted_medical_data')
    source_document = models.ForeignKey(
        MedicalDocument,
        on_delete=models.SET_NULL, # If original document is deleted, keep extracted data but clear link
        null=True,
        blank=True,
        help_text="Link to the original MedicalDocument this data was extracted from."
    )
    data_type_choices = [
        ('BLOOD_TEST', 'Blood Test Results'),
        ('DIAGNOSIS', 'Diagnosis'),
        ('MEDICATION', 'Medication List'),
        ('SYMPTOMS', 'Symptoms Description'),
        ('OTHER', 'Other Medical Information'),
    ]
    data_type = models.CharField(
        max_length=20,
        choices=data_type_choices,
        default='OTHER',
        help_text="Type of extracted medical data."
    )
    # JSONField to store the structured data (e.g., {'glucose': 120, 'unit': 'mg/dL', 'date': '2025-01-15'})
    extracted_json = models.JSONField(
        help_text="Structured medical data in JSON format."
    )
    extraction_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Extracted {self.data_type} for {self.user.username} on {self.extraction_date.strftime('%Y-%m-%d')}"

class MedicalAssistantInteraction(models.Model):
    """
    Logs the user's interaction with the AI assistant and the AI's final response.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='assistant_interactions')
    input_query = models.TextField(
        help_text="The initial query or context provided by the user that triggered the analysis."
    )
    # Store a summary of the historical context used for this interaction
    extracted_context_summary = models.JSONField(
        blank=True,
        null=True,
        help_text="Summary of historical data points used in this specific interaction."
    )
    reasoning_output = models.TextField(
        help_text="The raw analysis output from the Reasoning Model Agent."
    )
    final_analysis = models.TextField(
        help_text="The final, refined analysis output from the Revisor Agent, presented to the user."
    )
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Interaction for {self.user.username} at {self.timestamp.strftime('%Y-%m-%d %H:%M')}"