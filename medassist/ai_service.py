# AI service for medical assistant using OpenRouter API
import os
import requests
import json
from typing import Dict, Any, Optional
from django.conf import settings

class MedicalAIService:
    """
    Service class for handling AI interactions for medical analysis using OpenRouter
    """
    
    def __init__(self):
        # OpenRouter API configuration - read from environment variables
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Please set OPENROUTER_API_KEY in .env file.")
        
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:9002",  # Your app URL
            "X-Title": "Medical Assistant",  # Your app name
            "Content-Type": "application/json"
        }
        
        # You can choose different models available on OpenRouter
        # Popular options: openai/gpt-3.5-turbo, openai/gpt-4, anthropic/claude-3-haiku, etc.
        self.model = "openai/gpt-3.5-turbo"  # Change this to your preferred model
        
    def analyze_medical_text(self, text_input: str, user_history: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze medical text input using OpenRouter
        
        Args:
            text_input: User's medical question or description
            user_history: Optional previous medical history context
            
        Returns:
            Dict containing AI analysis and response
        """
        
        # Create a medical assistant prompt
        system_prompt = """
        You are an advanced medical AI assistant. Your role is to:
        1. Analyze medical symptoms, questions, and health concerns
        2. Provide helpful, accurate medical information
        3. Always recommend consulting with healthcare professionals
        4. Never provide definitive diagnoses
        5. Be empathetic and supportive
        
        Important: Always remind users that this is AI assistance and they should consult healthcare professionals for medical decisions.
        """
        
        user_prompt = f"""
        Medical inquiry: {text_input}
        
        {f"Previous medical context: {user_history}" if user_history else ""}
        
        Please provide a helpful, informative response while emphasizing the importance of professional medical consultation.
        """
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(self.base_url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            data = response.json()
            ai_response = data['choices'][0]['message']['content']
            
            return {
                "status": "success",
                "ai_response": ai_response,
                "model_used": self.model,
                "tokens_used": data.get('usage', {}).get('total_tokens', 0)
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "error": f"OpenRouter API request failed: {str(e)}",
                "ai_response": "I apologize, but I'm experiencing technical difficulties connecting to the AI service. Please try again later or consult with a healthcare professional."
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "ai_response": "I apologize, but I'm experiencing technical difficulties. Please try again later or consult with a healthcare professional."
            }
    
    def analyze_medical_document(self, document_text: str, document_type: str) -> Dict[str, Any]:
        """
        Analyze uploaded medical documents (OCR text from PDFs/images) using OpenRouter
        
        Args:
            document_text: Extracted text from medical document
            document_type: Type of document (PDF, IMAGE, etc.)
            
        Returns:
            Dict containing document analysis
        """
        
        system_prompt = """
        You are a medical document analysis AI. Your role is to:
        1. Extract and summarize key medical information from documents
        2. Identify important values, dates, and findings
        3. Explain medical terms in plain language
        4. Highlight any concerning values or recommendations
        5. Always emphasize the need for professional medical interpretation
        """
        
        user_prompt = f"""
        Please analyze this medical document text:
        
        Document Type: {document_type}
        Content: {document_text}
        
        Please provide:
        1. Summary of key findings
        2. Explanation of medical terms
        3. Any notable values or recommendations
        4. Reminder about professional medical consultation
        """
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 1200,
            "temperature": 0.6
        }
        
        try:
            response = requests.post(self.base_url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            data = response.json()
            ai_response = data['choices'][0]['message']['content']
            
            return {
                "status": "success",
                "ai_response": ai_response,
                "document_type": document_type,
                "model_used": self.model,
                "tokens_used": data.get('usage', {}).get('total_tokens', 0)
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "error": f"OpenRouter API request failed: {str(e)}",
                "ai_response": "I apologize, but I couldn't analyze the document due to connectivity issues. Please try again or consult with a healthcare professional."
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "ai_response": "I apologize, but I couldn't analyze the document. Please try again or consult with a healthcare professional."
            }