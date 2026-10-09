"""
Eiva agent - step 1: turn a command into an app name.

    "open telegram app"             ->  telegram
    "برنامه تلگرام رو برام باز کن"     ->  telegram

Run:  python eiva_agent.py      (type q to quit)
"""
import re
from AppOpener import open
import ollama

LANGUAGE_MODEL = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"

SYSTEM_PROMPT = (
    "You extract the application name from the user's command.\n"
    "Reply with ONLY the application name, in lowercase English letters.\n"
    "No punctuation, no explanation, no extra words.\n"
    "If the command does not mention an application, reply with exactly: none"
)

# Few-shot examples: a small model follows examples much better than rules.
# Add the apps YOU use (and the way you'd say them) to make it more reliable.
EXAMPLES = [
    ("open telegram app", "telegram"),
    ("برنامه تلگرام رو برام باز کن", "telegram"),
    ("can you launch chrome for me", "chrome"),
    ("اسپاتیفای رو اجرا کن", "spotify"),
    ("please start visual studio code", "visual studio code"),
    ("what's the weather today", "none"),
    ("سلام حالت چطوره", "none"),
]


def clean_output(text: str) -> str:
    """Keep only a short, lowercase app name, whatever extra text the model adds."""
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    name = re.sub(r"[^a-z0-9 .+#-]", "", first_line.lower()).strip()
    name = " ".join(name.split()[:4])   # app names are at most a few words
    return name or "none"


def extract_app_name(command: str) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_text, answer in EXAMPLES:
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": answer})
    messages.append({"role": "user", "content": command})

    response = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=messages,
        options={"temperature": 0, "num_predict": 12},  # deterministic and short
    )
    return clean_output(response["message"]["content"])


def main():
    while True:
        command = input("\nCommand: ").strip()
        if command.lower() == "q":
            break
        if not command:
            continue
        try:
            print(extract_app_name(command))
            open(extract_app_name(command))
        except (ollama.ResponseError, ConnectionError) as e:
            print(f"Ollama error: {e}")


if __name__ == "__main__":
    main()