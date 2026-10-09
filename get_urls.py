import requests
from bs4 import BeautifulSoup


def getdata(url):
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    r.encoding = "utf-8"
    return r.text


url = [
    "https://www.mongard.ir/articles/",
]

for i in range(2,19):
    url.append(f"https://www.mongard.ir/articles/?page={i}")

article_urls = []

for page_url in url:
    soup = BeautifulSoup(getdata(page_url), "html.parser")

    for div in soup.find_all("div", class_="card-body"):
        for a in div.find_all("a", href=True):
            link = a["href"]

            if link.startswith("/"):
                link = "https://www.mongard.ir" + link

            if link not in article_urls:
                article_urls.append(link)

print(article_urls)

with open("urls.txt", "w", encoding="utf-8") as f:
    for link in article_urls:
        f.write(link + "\n")