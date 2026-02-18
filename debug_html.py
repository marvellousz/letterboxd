import requests
from bs4 import BeautifulSoup
import cloudscraper
import json

# Try diary view which uses table format
url = "https://letterboxd.com/Marvellous_/films/diary/"

# Try cloudscraper first
scraper = cloudscraper.create_scraper(
    browser={'browser': 'chrome', 'platform': 'linux', 'mobile': False}
)

print(f"Testing URL: {url}")
response = scraper.get(url, timeout=20)
print(f"Status: {response.status_code}\n")

soup = BeautifulSoup(response.text, "html.parser")

# Check for table rows
table_rows = soup.select("tr.diary-entry-row, tr.film-row")
print(f"Table rows found: {len(table_rows)}")

if table_rows:
    print("\nFirst 3 table rows structure:")
    for i, row in enumerate(table_rows[:3]):
        print(f"\n{'='*60}")
        print(f"Row {i+1}:")
        print('='*60)
        
        # Get title - try multiple selectors
        title = None
        title_link = row.select_one("h3.headline-3 a, td.td-film-details h3 a, a.frame")
        if title_link:
            title = title_link.get_text(strip=True)
        
        # Try image alt text
        if not title:
            img = row.select_one("img")
            if img:
                title = img.get("alt", "")
        
        print(f"Title: {title or 'Unknown'}")
        
        # Look for rating with rated-X class
        rating_span = row.select_one("span.rating[class*='rated-']")
        if rating_span:
            classes = rating_span.get('class', [])
            for cls in classes:
                if cls.startswith('rated-'):
                    rating_num = cls.replace('rated-', '')
                    print(f"Rating: {rating_num}/10 (class: {cls})")
                    break
        
        # Look for year
        year = None
        year_link = row.select_one("td.td-released a, td.td-film-details small a")
        if year_link:
            year_text = year_link.get_text(strip=True)
            print(f"Year: {year_text}")
else:
    print("No table rows found - diary view might be empty")
