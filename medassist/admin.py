from django.contrib import admin
from .models import MedicalDocument, ExtractedMedicalData, MedicalAssistantInteraction

@admin.register(MedicalDocument)
class MedicalDocumentAdmin(admin.ModelAdmin):
    list_display = ('user', 'document_type', 'upload_date', 'file')
    list_filter = ('document_type', 'upload_date')
    search_fields = ('user__username', 'raw_text')
    readonly_fields = ('upload_date',)

@admin.register(ExtractedMedicalData)
class ExtractedMedicalDataAdmin(admin.ModelAdmin):
    list_display = ('user', 'data_type', 'extraction_date', 'source_document')
    list_filter = ('data_type', 'extraction_date')
    search_fields = ('user__username',)
    readonly_fields = ('extraction_date',)

@admin.register(MedicalAssistantInteraction)
class MedicalAssistantInteractionAdmin(admin.ModelAdmin):
    list_display = ('user', 'timestamp', 'input_preview')
    list_filter = ('timestamp',)
    search_fields = ('user__username', 'input_query')
    readonly_fields = ('timestamp',)
    
    def input_preview(self, obj):
        return obj.input_query[:50] + "..." if len(obj.input_query) > 50 else obj.input_query
    input_preview.short_description = 'Input Preview'
