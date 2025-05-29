from django.contrib import admin
from .models import MedicalDocument, ChatMessage

@admin.register(MedicalDocument)
class MedicalDocumentAdmin(admin.ModelAdmin):
    list_display = ('user', 'file_type', 'uploaded_at', 'file')
    list_filter = ('file_type', 'uploaded_at')
    search_fields = ('user__username', 'file__name')
    readonly_fields = ('uploaded_at',)

@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ('user', 'sender', 'timestamp', 'message_preview')
    list_filter = ('sender', 'timestamp')
    search_fields = ('user__username', 'message')
    readonly_fields = ('timestamp',)
    
    def message_preview(self, obj):
        return obj.message[:50] + "..." if len(obj.message) > 50 else obj.message
    message_preview.short_description = 'Message Preview'
