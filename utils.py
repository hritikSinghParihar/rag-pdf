import os
import logging
from typing import List, Tuple
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

def is_pdf_scanned(pdf_path: str, text_threshold: int = 100) -> bool:
    """
    Check if a PDF is scanned (contains mostly images/path objects) 
    or has extractable text.
    """
    try:
        doc = fitz.open(pdf_path)
        total_text = ""
        for page in doc:
            total_text += page.get_text()
        
        # If total text is very short across all pages, it's likely scanned
        return len(total_text.strip()) < text_threshold
    except Exception as e:
        logger.error(f"Error checking PDF type: {e}")
        return True # Default to OCR if error

def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[str]:
    """Convert PDF pages to temporary image files for OCR."""
    try:
        images = convert_from_path(pdf_path, dpi=dpi)
        image_paths = []
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        temp_dir = "temp_ocr"
        os.makedirs(temp_dir, exist_ok=True)
        
        for i, image in enumerate(images):
            path = os.path.join(temp_dir, f"{base_name}_page_{i+1}.png")
            image.save(path, "PNG")
            image_paths.append(path)
        
        return image_paths
    except Exception as e:
        logger.error(f"Error converting PDF to images: {e}")
        return []

def preprocess_image(image_path: str) -> str:
    """Basic image preprocessing to improve OCR accuracy."""
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return image_path
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Optional: Binarization (Otsu's thresholding)
        # _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        processed_path = image_path.replace(".png", "_processed.png")
        cv2.imwrite(processed_path, denoised)
        return processed_path
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return image_path

def get_all_supported_files(directory: str) -> List[str]:
    """Recursively find all supported files (PDF, TXT, HTML, DOCX, Images) in a directory."""
    supported_extensions = ('.pdf', '.txt', '.html', '.htm', '.docx', '.png', '.jpg', '.jpeg')
    file_paths = []
    if not os.path.exists(directory):
        logger.warning(f"Directory not found: {directory}")
        return []
        
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(supported_extensions):
                file_paths.append(os.path.join(root, file))
    return file_paths
