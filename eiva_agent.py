"""
Eiva Agent - a tiny window: type a command, the LLM extracts the app name,
and the app is opened.

    "open telegram app"             ->  telegram  ->  opens Telegram
    "برنامه تلگرام رو برام باز کن"     ->  telegram  ->  opens Telegram

Install:  pip install customtkinter ollama AppOpener
Run:      python eiva_agent.py
"""
import queue
import re
import threading

import customtkinter as ctk
import ollama
from AppOpener import open as open_app   # renamed so it doesn't shadow Python's built-in open()

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


class AgentWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.title("Eiva Agent")
        self.geometry("460x240")
        self.resizable(False, False)

        self.events: queue.Queue = queue.Queue()

        ctk.CTkLabel(self, text="⚡  Eiva Agent", font=ctk.CTkFont(size=20, weight="bold")
                     ).pack(anchor="w", padx=22, pady=(18, 0))
        ctk.CTkLabel(self, text="Tell me which app to open.", text_color="#8a8fa3",
                     font=ctk.CTkFont(size=12)).pack(anchor="w", padx=22)

        self.entry = ctk.CTkEntry(self, height=44, corner_radius=22, font=ctk.CTkFont(size=14),
                                  placeholder_text="e.g. open telegram")
        self.entry.pack(fill="x", padx=22, pady=(16, 8))
        self.entry.bind("<Return>", lambda e: self.run())
        self.entry.focus()

        self.button = ctk.CTkButton(self, text="Run", height=40, corner_radius=20, command=self.run)
        self.button.pack(fill="x", padx=22)

        self.status = ctk.CTkLabel(self, text="", text_color="#8a8fa3", wraplength=410,
                                   justify="left", font=ctk.CTkFont(size=12))
        self.status.pack(anchor="w", padx=22, pady=12)

        self.after(50, self._poll)

    def run(self):
        command = self.entry.get().strip()
        if not command or self.button.cget("state") == "disabled":
            return
        self.button.configure(state="disabled")
        self.status.configure(text="Thinking…")
        threading.Thread(target=self._work, args=(command,), daemon=True).start()

    def _work(self, command: str):
        """Runs in a background thread so the window never freezes."""
        try:
            name = extract_app_name(command)       # one LLM call only
            if name == "none":
                self.events.put(("status", "I couldn't find an app name in that command."))
                return
            self.events.put(("status", f"Opening {name}…"))
            open_app(name, match_closest=True, output=False, throw_error=True)
            self.events.put(("status", f"✓ {name}"))
        except (ollama.ResponseError, ConnectionError) as e:
            self.events.put(("status", f"Ollama error: {e}"))
        except Exception as e:  # thread boundary: AppOpener raises when the app isn't found
            self.events.put(("status", f"✗ Couldn't open the app: {e}"))
        finally:
            self.events.put(("idle", None))

    def _poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "status":
                    self.status.configure(text=value)
                elif kind == "idle":
                    self.button.configure(state="normal")
                    self.entry.delete(0, "end")
                    self.entry.focus()
        except queue.Empty:
            pass
        self.after(50, self._poll)


if __name__ == "__main__":
    AgentWindow().mainloop()