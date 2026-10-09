import sys
import webbrowser
from urllib.parse import quote_plus


def main():
    if len(sys.argv) < 2:
        print('Usage: python eiva_search.py "search input"')
        sys.exit(1)

    # اگه کاربر بدون کوتیشن چندتا کلمه نوشت، همه رو بچسبون به هم
    query = " ".join(sys.argv[1:])
    url = f"https://www.google.com/search?q={quote_plus(query)}"

    try:
        chrome = webbrowser.get("chrome")
    except webbrowser.Error:
        print("Chrome پیدا نشد؛ با مرورگر پیش‌فرض باز می‌شود.")
        chrome = webbrowser

    chrome.open(url)


if __name__ == "__main__":
    main()