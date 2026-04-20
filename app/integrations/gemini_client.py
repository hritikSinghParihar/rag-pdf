import google.generativeai as genai
from app.core.config import settings

class GeminiClient:
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(settings.GEMINI_MODEL)

    def generate_content(self, prompt: str, image: any = None) -> str:
        if image:
            response = self.model.generate_content([prompt, image])
        else:
            response = self.model.generate_content(prompt)
        return response.text

gemini_client = GeminiClient()
