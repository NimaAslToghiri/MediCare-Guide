from django.db import models
from django.contrib.auth.models import User

class MedicalDocument(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    file = models.FileField(upload_to='documents/')
    file_type = models.CharField(max_length=20, choices=[
        ('pdf', 'PDF'),
        ('image', 'Image'),
        ('text', 'Plain Text')
    ])

    def __str__(self):
        return f"{self.user.username} - {self.file.name}"


class ChatMessage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    sender = models.CharField(max_length=10, choices=[('user', 'User'), ('agent', 'Agent')], default='agent')
    message = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.sender} @ {self.timestamp}: {self.message[:30]}"
