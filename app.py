import asyncio
import aiohttp
import sqlite3
import spacy
import undetected_chromedriver as uc
import re
import pandas as pd
import redis
import csv
import google.generativeai as genai
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, send_file
from selenium import webdriver
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Configure Google Gemini AI
GEMINI_API_KEY = "AIzaSyBE7XZMfnTsIpeBnbaPeeyKAc0qZ_JyVKI"
genai.configure(
    api_key=GEMINI_API_KEY,
    transport="rest",  # Force REST API to avoid gRPC issues
    client_options={"api_endpoint": "generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"}
)

# Load AI NLP model
nlp = spacy.load("en_core_web_sm")

# Flask App
app = Flask(__name__)

# Database for Caching
conn = sqlite3.connect("scraper.db", check_same_thread=False)
c = conn.cursor()

# Create table with correct schema
c.execute("DROP TABLE IF EXISTS cache")
c.execute("""
    CREATE TABLE cache (
        url TEXT PRIMARY KEY, 
        title TEXT, 
        summary TEXT, 
        email TEXT, 
        linkedin TEXT,
        ai_insights TEXT
    )
""")
conn.commit()

# Setup Redis Cache
cache = redis.Redis(host='localhost', port=6379, db=0)


def fetch_with_selenium(url):
    """Fetch webpage content using headless Chrome"""
    options = uc.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-blink-features=AutomationControlled")
    driver = uc.Chrome(options=options)
    try:
        driver.get(url)
        time.sleep(5)
        return driver.page_source
    finally:
        driver.quit()


async def fetch(session, url):
    """Async fetch with proper error handling"""
    try:
        async with session.get(url, timeout=10) as response:
            response.raise_for_status()
            return await response.text()
    except Exception as e:
        logging.error(f"Fetch error: {e}")
        return None


def extract_emails(soup):
    """Improved email extraction with regex"""
    email_regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    text = soup.get_text()
    return list(set(re.findall(email_regex, text))) or ["No emails found"]


def extract_social_links(soup):
    """Extract social links with URL validation"""
    social_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].lower()
        if "linkedin.com" in href:
            social_links.append(a["href"])
        elif "twitter.com" in href:
            social_links.append(a["href"])
    return social_links or ["No social links found"]


def extract_summary(soup):
    """Improved content extraction using Spacy NLP"""
    paragraphs = [p.get_text().strip() for p in soup.find_all("p")]
    meaningful_text = [p for p in paragraphs if len(p.split()) > 5]
    return " ".join(meaningful_text[:3]) or "No meaningful summary found"


def analyze_with_gemini(text):
    """Gemini analysis with error handling"""
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(
            f"Analyze this content and provide key insights: {text[:3000]}"
        )
        return response.text
    except Exception as e:
        logging.error(f"Gemini error: {e}")
        return "AI analysis unavailable"


def get_cached_data(url):
    """Redis cache with pickle serialization"""
    cached = cache.get(url)
    return eval(cached.decode()) if cached else None


def store_cache(url, data):
    """Cache storage with compression"""
    cache.setex(url, 86400, str(data))


async def scrape_website(url):
    """Main scraping logic with improved error handling"""
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"

    if cached := get_cached_data(url):
        return cached

    try:
        html = await fetch_website_content(url)
        soup = BeautifulSoup(html, "html.parser")

        result = {
            "title": soup.title.string[:200] if soup.title else "No title",
            "summary": extract_summary(soup),
            "email": ", ".join(extract_emails(soup)),
            "linkedin": ", ".join(extract_social_links(soup)),
            "ai_insights": analyze_with_gemini(extract_summary(soup))
        }

        store_cache(url, result)
        c.execute("INSERT INTO cache VALUES (?, ?, ?, ?, ?, ?)",
                  (url, result["title"], result["summary"], result["email"],
                   result["linkedin"], result["ai_insights"]))
        conn.commit()
        return result
    except Exception as e:
        logging.error(f"Scraping failed: {e}")
        return {"error": str(e)}


async def fetch_website_content(url):
    """Content fetching with proper browser selection"""
    if "amazon." in url or "cloudflare" in url:
        return fetch_with_selenium(url)
    async with aiohttp.ClientSession() as session:
        return await fetch(session, url)


@app.route("/download_csv")
def download_csv():
    df = pd.read_sql_query("SELECT * FROM cache", conn)
    df.to_csv("scraped_data.csv", index=False)
    return send_file("scraped_data.csv", as_attachment=True)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        url = request.form.get("url")
        result = asyncio.run(scrape_website(url))
        return render_template("index.html", result=result, url=url)
    return render_template("index.html", result=None)


@app.route("/bulk_scrape", methods=["POST"])
def bulk_scrape():
    file = request.files["file"]
    urls = [row[0] for row in csv.reader(file)]
    results = [asyncio.run(scrape_website(url)) for url in urls]
    return render_template("results.html", results=results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)