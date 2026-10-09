import re
import requests
from bs4 import BeautifulSoup


def getdata(url):
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    r.encoding = "utf-8"
    return r.text

url = "https://www.mongard.ir/articles/"
soup = BeautifulSoup(getdata(url), "html.parser")

article_urls = []

for div in soup.find_all("div", class_="card-body"):
    for a in div.find_all("a"):
        article_urls.append(a["href"])

print(article_urls)

with open("urls.txt", "w", encoding="utf-8") as f:
    for link in article_urls:
        f.write(f"https://www.mongard.ir{link}" + "\n")

