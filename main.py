import os
import re
import time
import requests
from bs4 import BeautifulSoup

HEADERS = {"Use r-Agent": "Mozilla/5.0"}
OUTPUT_DIR = "articles"


def getdata(url):
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    r.encoding = "utf-8"
    return r.text


def safe_filename(name: str) -> str:
    # حذف کاراکترهای غیرمجاز در ویندوز
    name = re.sub(r'[\\/:*?"<>|؟]', "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:150] or "untitled"


def scrape_article(url):
    soup = BeautifulSoup(getdata(url), "html.parser")

    title = soup.title.text.strip() if soup.title else "untitled"

    content = soup.find("article", class_="blog-content-wrapper")
    if not content:
        print(f"[!] محتوا پیدا نشد: {url}")
        return

    paragraphs = [p.text.strip() for p in content.find_all("p") if p.text.strip()]
    paragraphs = list(dict.fromkeys(paragraphs))  # حذف پاراگراف‌های تکراری

    filename = os.path.join(OUTPUT_DIR, f"{safe_filename(title)}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(title + "\n\n")
        f.write("\n\n".join(paragraphs))

    print(f"[+] ذخیره شد: {filename}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open("urls.txt", "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    for i, link in enumerate(urls, start=1):
        print(f"({i}/{len(urls)}) {link}")
        try:
            scrape_article(link)
        except requests.RequestException as e:
            print(f"[x] خطا در دریافت {link}: {e}")
        time.sleep(1)


if __name__ == "__main__":
    main()