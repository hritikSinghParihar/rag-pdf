import base64
import json
import logging
import time
from typing import Dict, Any, Optional
import os

from openai import OpenAI
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config import config

logger = logging.getLogger(__name__)

def encode_image(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class VisionOCR:
    def __init__(self, provider: str = None):
        self.provider = provider or config.provider
        self.openai_client = None
        if self.provider == "openai" and config.openai_api_key:
            self.openai_client = OpenAI(api_key=config.openai_api_key)
        elif self.provider == "gemini" and config.gemini_api_key:
            genai.configure(api_key=config.gemini_api_key)
            self.gemini_model = genai.GenerativeModel(config.gemini_model)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((Exception)), # Broad for now, can be narrowed
        reraise=True
    )
    def extract_structured_data(self, image_path: str) -> Dict[str, Any]:
        """Extract text and layout structure from image using Vision models."""
        prompt = """
        Analyze this image and extract all text content. 
        Identify the document structure including:
        - Main Title
        - Headings
        - Sections/Paragraphs
        - Tables (if any, as structured data)
        
        Return the result ONLY as a valid JSON object with the following schema:
        {
          "title": "Document Title",
          "sections": [
            {
              "heading": "Heading Name",
              "content": "Paragraph text...",
              "type": "text | table",
              "table_data": [[row1_col1, row1_col2], ...] # Only if type is table
            }
          ],
          "metadata": {
            "language": "en",
            "page_number": 1
          }
        }
        """
        
        try:
            if self.provider == "openai" and self.openai_client:
                return self._extract_openai(image_path, prompt)
            elif self.provider == "gemini":
                return self._extract_gemini(image_path, prompt)
            else:
                raise ValueError(f"Provider {self.provider} not configured or supported.")
        except Exception as e:
            logger.error(f"Vision OCR error on {image_path}: {e}")
            raise

    def _extract_openai(self, image_path: str, prompt: str) -> Dict[str, Any]:
        base64_image = encode_image(image_path)
        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        },
                    ],
                }
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

    def _extract_gemini(self, image_path: str, prompt: str) -> Dict[str, Any]:
        # Implementation for Gemini (using PIL Image for better compatibility)
        from PIL import Image
        img = Image.open(image_path)
        response = self.gemini_model.generate_content([prompt, img])
        content = response.text
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        return json.loads(content)
