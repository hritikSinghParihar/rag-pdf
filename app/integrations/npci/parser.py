from typing import List, Dict, Any

def parse_npci_press_releases(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Parses press release items from custom NPCI API response (data.results)."""
    items = []
    # Structure for press-release-details endpoint
    results = data.get("data", {}).get("results", [])
    if not results:
        # Fallback to collection structure
        results = data.get("data", [])
        
    if isinstance(results, list):
        for entry in results:
            attr = entry.get("attributes", entry) 
            title = attr.get("title")
            
            # Check multiple possible paths for media/pdf URL
            media = attr.get("media", {})
            pdf = attr.get("pdf", {})
            
            url = None
            # Path 1: media.url (flat)
            if isinstance(media, dict) and media.get("url"):
                url = media.get("url")
            # Path 2: media.data.attributes.url (nested)
            elif isinstance(media, dict) and media.get("data"):
                url = media.get("data", {}).get("attributes", {}).get("url")
            # Path 3: pdf.url (flat)
            elif isinstance(pdf, dict) and pdf.get("url"):
                url = pdf.get("url")
            # Path 4: pdf.data.attributes.url (nested)
            elif isinstance(pdf, dict) and pdf.get("data"):
                url = pdf.get("data", {}).get("attributes", {}).get("url")
            # Path 5: entry.url (direct)
            elif attr.get("url"):
                url = attr.get("url")
            
            if title and url:
                items.append({
                    "name": title,
                    "url": url,
                    "type": "press_release"
                })
    return items

def parse_npci_media_coverages(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Parses media coverage items from Strapi API response."""
    items = []
    data_list = data.get("data", [])
    if not isinstance(data_list, list):
        return items

    for entry in data_list:
        attr = entry.get("attributes", {})
        details = attr.get("details", [])
        if isinstance(details, list):
            for detail in details:
                if detail.get("mediaType") == "pdf":
                    title = detail.get("title")
                    
                    media = detail.get("media", {})
                    pdf = detail.get("pdf", {})
                    
                    url = None
                    if isinstance(media, dict) and media.get("url"):
                        url = media.get("url")
                    elif isinstance(media, dict) and media.get("data"):
                        url = media.get("data", {}).get("attributes", {}).get("url")
                    elif isinstance(pdf, dict) and pdf.get("url"):
                        url = pdf.get("url")
                    elif isinstance(pdf, dict) and pdf.get("data"):
                        url = pdf.get("data", {}).get("attributes", {}).get("url")
                        
                    if title and url:
                        items.append({
                            "name": title,
                            "url": url,
                            "type": "media_coverage"
                        })
    return items
