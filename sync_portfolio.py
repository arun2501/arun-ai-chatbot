"""
sync_portfolio.py
Fetches recently updated information directly from Arunkumar's live portfolio
(https://krarun-portfolio.netlify.app/) and updates the RAG knowledge base.
"""

import os
import json
import time
import re
import urllib.request
import urllib.error
from datetime import datetime

PORTFOLIO_URL = "https://krarun-portfolio.netlify.app/"
SUBPAGES = [
    "",
    "portfolio-details",      # Jewelry UI/UX
    "portfolio-details1",     # Nebulizer
    "porfolio-detailsweb",    # Assets Management System
    "portfolio-detailsiee",   # Fruit Health MMCNN
    "portfolio-detailsieee1", # Smart e-PDS
    "portfoilo-detailspds"    # Online e-PDS
]

def clean_html_text(html_text: str) -> str:
    """Strip HTML tags and clean whitespace."""
    # Remove script and style tags
    cleaned = re.sub(r'<(script|style)[^>]*>[\s\S]*?</\1>', ' ', html_text, flags=re.IGNORECASE)
    # Remove HTML tags
    cleaned = re.sub(r'<[^>]+>', ' ', cleaned)
    # Decode HTML entities
    cleaned = cleaned.replace('&amp;', '&').replace('&nbsp;', ' ').replace('&quot;', '"').replace('&#39;', "'")
    # Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def fetch_portfolio_data():
    """Fetch live data from portfolio and return structured info."""
    synced_items = []
    base_dir = os.path.dirname(__file__)
    knowledge_dir = os.path.join(base_dir, "knowledge")
    os.makedirs(knowledge_dir, exist_ok=True)
    out_file = os.path.join(knowledge_dir, "live_portfolio.json")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting portfolio sync from {PORTFOLIO_URL}...")

    for page in SUBPAGES:
        target_url = f"{PORTFOLIO_URL.rstrip('/')}/{page.lstrip('/')}"
        try:
            req = urllib.request.Request(target_url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                content = response.read().decode('utf-8', errors='ignore')
                raw_text = clean_html_text(content)
                title_match = re.search(r'<title>(.*?)</title>', content, re.IGNORECASE)
                title = title_match.group(1).strip() if title_match else page or "Home"

                synced_items.append({
                    "url": target_url,
                    "title": title,
                    "page_slug": page or "home",
                    "text_preview": raw_text[:4000], # store clean textual content for RAG chunks
                    "synced_at": datetime.now().isoformat()
                })
                print(f"  [OK] Synced {target_url} ({len(raw_text)} chars)")
        except Exception as e:
            print(f"  [WARN] Could not fetch {target_url}: {e}")

    result_payload = {
        "source": PORTFOLIO_URL,
        "last_synced": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_pages": len(synced_items),
        "pages": synced_items
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2, ensure_ascii=False)

    print(f"Portfolio sync complete! Saved {len(synced_items)} pages to {out_file}")
    return result_payload

if __name__ == "__main__":
    fetch_portfolio_data()
