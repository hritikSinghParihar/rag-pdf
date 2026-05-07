from selectolax.parser import HTMLParser
from urllib.parse import urljoin
from typing import List, Dict, Any
import logging

logger = logging.getLogger("rag_app")

def parse_hidden_form_data(html: str) -> Dict[str, str]:
    """Parses hidden input fields from ASP.NET form."""
    tree = HTMLParser(html)
    return {
        tag.attributes.get("name"): tag.attributes.get("value", "") 
        for tag in tree.css("input[type='hidden']") 
        if tag.attributes.get("name")
    }

def parse_document_links(html: str, base_url: str, doc_type: str = "circular") -> List[Dict[str, str]]:
    """Extracts document links and titles from the index/notification page."""
    tree = HTMLParser(html)
    links = {}
    
    # Base paths for different document types
    base_paths = {
        "notification": "NotificationUser.aspx",
        "circular": "BS_CircularIndexDisplay.aspx",
        "master_direction": "BS_ViewMasDirections.aspx",
        "master_circular": "BS_ViewMasterCirculars.aspx"
    }
    
    base_path = base_paths.get(doc_type, "BS_CircularIndexDisplay.aspx")
        
    for a in tree.css("a"):
        href = a.attributes.get("href", "")
        # Match links that have an ID (periodic circulars) or are internal script links
        if f"{base_path}?Id=" in href or f"{base_path}?id=" in href:
            full_url = urljoin(base_url, href)
            name = a.text(strip=True) or (f"{doc_type.capitalize()}_" + href.split("d=")[-1])
            links[full_url] = {"url": full_url, "name": name}
    
    return list(links.values())

def parse_pdf_link(html: str, base_url: str) -> str:
    """Extracts the direct PDF download link from a detail page."""
    tree = HTMLParser(html)
    # RBI often uses an <a> tag with id starting with APDF_ for the PDF link
    pdf_tag = tree.css_first('a[id^="APDF_"]')
    
    if not pdf_tag:
        # Fallback: look for any link ending in .PDF
        for a in tree.css("a"):
            href = a.attributes.get("href", "")
            if href.lower().endswith(".pdf"):
                pdf_tag = a
                break
                
    if not pdf_tag:
        return ""
    
    pdf_url = pdf_tag.attributes.get("href", "")
    if not pdf_url.startswith('http'):
        pdf_url = urljoin(base_url, pdf_url)
    return pdf_url
