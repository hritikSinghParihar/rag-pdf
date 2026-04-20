import os
from typing import Optional
from openai import OpenAI
from app.core.config import settings

class OpenAIClient:
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        self.client = None
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)

    def generate_chat_completion(self, messages: list, model: str = None) -> Optional[str]:
        if not self.client:
            return None
        model = model or settings.OPENAI_MODEL
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )
        return response.choices[0].message.content

openai_client = OpenAIClient()
