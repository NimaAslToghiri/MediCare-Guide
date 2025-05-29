from .models import MedicalDocument
import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')


def reasoning_agent(user_input, history_summary):
    prompt = f"""
You're a professional AI doctor assistant. A user has provided the following medical input or description:

--- Start of User Input ---
{user_input}
--- End of User Input ---

Here is some relevant history from this user:
{history_summary}

Please provide a clear medical reasoning or analysis based on this information. Be informative and structured. 
If you need more information from the user, ask specific follow-up questions.
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a medical reasoning assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI API error in reasoning_agent: {e}")
        return "I'm having trouble processing your request right now. Please try again later."


def revisor_agent(reasoning_output, user_input, history_summary):
    prompt = f"""
You're a quality assurance assistant for a medical AI system.

The following is the user's input:
{user_input}

This is the medical history summary:
{history_summary}

This is the AI-generated response to review:
--- Start of Reasoning Output ---
{reasoning_output}
--- End of Reasoning Output ---

Please revise or enhance this response to improve its clarity, completeness, and usefulness.
If the reasoning is already good, keep it mostly intact but polish the language.
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a medical reasoning reviewer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI API error in revisor_agent: {e}")
        return reasoning_output  # Return original reasoning if revision fails


def input_validator_agent(user_input: str) -> bool:
    """
    Returns True if user input is relevant to medical context, False otherwise.
    """
    prompt = f"""
You are a helpful assistant in a medical document analysis system.

Determine if the following input is relevant to the task of helping users with their medical documents (e.g., blood test results, medical reports, health-related questions).

Respond ONLY with 'yes' or 'no'.

Input: "{user_input}"
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a relevance classifier for medical assistant input."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )
        answer = response.choices[0].message.content.strip().lower()
        return 'yes' in answer
    except Exception as e:
        print(f"OpenAI API error in input_validator_agent: {e}")
        return True  # Default to allowing input if validation fails


def intro_agent(user):
    history = MedicalDocument.objects.filter(user=user).order_by('-uploaded_at')

    if history.exists():
        last_doc = history.first()
        return f"""
        👋 Welcome back, {user.username}!

        I see your last uploaded file was: **{last_doc.file.name}** on {last_doc.uploaded_at.date()}.
        I'm ready to analyze more data or continue from where we left off.
        Please upload another report if you'd like.
        """
    else:
        return f"""
        👋 Hello {user.username}, and welcome to your personal doctor assistant!

        Please upload your first medical report (PDF, image, or plain text).
        I'll help analyze it and provide a smart summary. 😊
        """


def intro_agent_with_llm(user):
    history_summary = get_user_history_summary(user)

    prompt = f"""
You are a helpful, friendly medical assistant agent.
A user named {user.username} just logged in.

Their document history:
{history_summary}

Based on that, greet the user and explain how they can upload a new report.
Use friendly, professional tone.
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a doctor assistant chatbot."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API error in intro_agent_with_llm: {e}")
        return intro_agent(user)  # Fallback to simple intro


def get_user_history_summary(user):
    docs = MedicalDocument.objects.filter(user=user).order_by('-uploaded_at')
    if not docs.exists():
        return "User has no prior medical documents."

    summaries = []
    for doc in docs[:3]:  # Limit to last 3 documents
        summaries.append(f"- {doc.file.name} uploaded on {doc.uploaded_at.date()}")
    return "\n".join(summaries)


def generate_agent_reply(context):
    prompt = f"""
You're a medical assistant helping a user analyze their medical documents and history.
Below is the conversation so far:

{context}

Now, based on the above, continue the conversation. Be informative, focused on health-related tasks.
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a health-focused assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API error in generate_agent_reply: {e}")
        return "I'm sorry, I'm having trouble generating a response right now. Please try again."
