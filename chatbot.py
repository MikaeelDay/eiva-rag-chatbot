"""
Eiva - RAG chatbot with a customtkinter UI (English interface, English answers)

Install:
    pip install customtkinter ollama numpy
    ollama pull bge-m3            # multilingual embedding model (reads the Persian articles)

Folder layout:
    eiva_gui.py
    articles/*.txt
"""
import hashlib
import json
import queue
import re
import threading
import tkinter as tk
from dataclasses import dataclass, asdict
from pathlib import Path
import subprocess
import sys
import customtkinter as ctk
import numpy as np
import ollama

# ───────────────────────── Settings ─────────────────────────
ARTICLES_DIR = Path("./articles")
CACHE_FILE = Path("./.eiva_cache.json")

EMBEDDING_MODEL = "bge-m3"
# A 1B model is weak at reading Persian context. If you can, use a bigger one,
# e.g. "qwen2.5:7b" or "llama3.2:3b" (ollama pull <name> first).
LANGUAGE_MODEL = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"

BOT_NAME = "eiva"
CHUNK_SIZE = 900        # max characters per chunk
TOP_N = 4               # chunks given to the model
MIN_SIMILARITY = 0.30   # below this = "not in the articles". Lower it if too strict.
EMBED_BATCH = 16
HISTORY_TURNS = 6       # previous messages the model can see
SCROLL_STEP = 4         # pixels per scroll unit: one wheel notch = 20 units = 80 px (raise = faster)
PAD_Y = 10              # vertical padding inside message bubbles

NOT_FOUND = "I couldn't find anything about that in the articles. Could you rephrase your question?"

# English display names for the sidebar (key = file name without ".txt" and
# without the "مونگارد " prefix). Files not listed here show their file name.
TITLE_MAP = {
    "آموزش logging در جنگو": "Logging in Django",
    "آموزش نصب pycharm در ویندوز": "Installing PyCharm on Windows",
    "آموزش نصب پکیج ها در زمان قطعی اینترنت": "Installing packages without internet",
    "آموزش کامل متغیر ثابت در پایتون": "Constants in Python",
    "اجرای سلری در ویندوز": "Running Celery on Windows",
    "تابع setdefault پایتون": "Python setdefault()",
    "خطای builtin_function_or_method object is not subscriptable در پایتون":
        "'builtin_function_or_method' is not subscriptable",
    "دریافت آخرین آیتم یک لیست در پایتون": "Getting the last item of a list",
    "ساخت فایل های pdf در جنگو": "Creating PDF files in Django",
    "عملیات ضرب در پایتون": "Multiplication in Python",
    "پیشنهاد بهبود پایتون یا PEP چیست": "What is a PEP?",
    "گرد کردن اعداد با تابع round پایتون": "Rounding numbers with round()",
}

FENCE = "`" * 3
CODE_RE = re.compile(FENCE + r"([\w+#.-]*)[ \t]*\n(.*?)(?:" + FENCE + r"|\Z)", re.S)
INLINE_RE = re.compile(r"(`[^`\n]+`|\*\*[^*\n]+\*\*)")
NOISE_RE = re.compile(r"^(ویدیو پیشنهادی|دوره پیشنهادی)\s*:")

# Colors
C_SIDEBAR = "#14161d"
C_MAIN = "#0f1117"
C_BOT = "#1a1e29"
C_USER = "#2f4fa8"
C_ACCENT = "#2a3a6b"
C_HOVER = "#232838"


# ───────────────────────── Article processing ─────────────────────────
@dataclass
class Chunk:
    title: str
    text: str


def clean_article(text: str) -> list[str]:
    """Drop duplicated paragraphs and the 'suggested video/course' ad lines."""
    seen, out = set(), []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para or NOISE_RE.match(para):
            continue
        key = re.sub(r"\s+", " ", para)
        if len(key) > 40:  # only dedupe long paragraphs (short code lines like "}" must stay)
            if key in seen:
                continue
            seen.add(key)
        out.append(para)
    return out


def chunk_paragraphs(paras: list[str], size: int) -> list[str]:
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) > size:
            chunks.append(cur)
            cur = ""
        cur += ("\n" if cur else "") + p
    if cur:
        chunks.append(cur)
    return chunks


class KnowledgeBase:
    def __init__(self):
        self.files = sorted(ARTICLES_DIR.glob("*.txt"))
        self.chunks: list[Chunk] = []
        self.matrix = None

    @staticmethod
    def title_of(path: Path) -> str:
        stem = path.stem.strip()
        stem = re.sub(r"^مونگارد\s*[|:-]?\s*", "", stem) or stem
        return TITLE_MAP.get(stem, stem)

    def article_text(self, path: Path) -> str:
        raw = path.read_text(encoding="utf-8-sig")
        return "\n".join(clean_article(raw))

    def _signature(self) -> str:
        info = [(p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in self.files]
        raw = json.dumps([EMBEDDING_MODEL, CHUNK_SIZE, TITLE_MAP, info], ensure_ascii=False)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _set_matrix(self, embeddings):
        m = np.array(embeddings, dtype=np.float32)
        m /= np.linalg.norm(m, axis=1, keepdims=True)
        self.matrix = m

    def build(self, progress):
        sig = self._signature()
        if CACHE_FILE.exists():
            try:
                data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
                if data.get("sig") == sig:
                    self.chunks = [Chunk(**c) for c in data["chunks"]]
                    self._set_matrix(data["embeddings"])
                    progress(1, 1)
                    return
            except (OSError, ValueError, KeyError, TypeError):
                pass  # broken cache, rebuild

        chunks = []
        for p in self.files:
            try:
                paras = clean_article(p.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeError):
                continue
            title = self.title_of(p)
            for part in chunk_paragraphs(paras, CHUNK_SIZE):
                chunks.append(Chunk(title, part))
        if not chunks:
            raise RuntimeError("No articles found in the 'articles' folder.")

        embeddings = []
        for i in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[i:i + EMBED_BATCH]
            res = ollama.embed(model=EMBEDDING_MODEL,
                               input=[f"{c.title}\n{c.text}" for c in batch])
            embeddings.extend(res["embeddings"])
            progress(len(embeddings), len(chunks))

        self.chunks = chunks
        self._set_matrix(embeddings)
        try:
            CACHE_FILE.write_text(
                json.dumps({"sig": sig, "chunks": [asdict(c) for c in chunks],
                            "embeddings": embeddings}, ensure_ascii=False),
                encoding="utf-8")
        except OSError:
            pass

    def retrieve(self, query: str, top_n: int = TOP_N):
        q = np.array(ollama.embed(model=EMBEDDING_MODEL, input=query)["embeddings"][0],
                     dtype=np.float32)
        q /= np.linalg.norm(q)
        sims = self.matrix @ q
        idx = np.argsort(-sims)[:top_n]
        return [(self.chunks[i], float(sims[i])) for i in idx]


def build_messages(question, hits, history):
    context = "\n\n".join(f"[{i}] ({c.title})\n{c.text}" for i, (c, _) in enumerate(hits, 1))
    system = (
        f"You are {BOT_NAME},if asked your name you answer {BOT_NAME}, a friendly assistant that answers questions about programming articles.\n"
        "Rules:\n"
        "- Answer ONLY from the context below. If the answer is not in the context, say you could not find it in the articles.\n"
        "- The context is mostly written in Persian. Read it, but ALWAYS write your answer in English.\n"
        "- Keep code, identifiers and technical terms exactly as they appear in the context.\n"
        "- Keep the answer short and conversational.\n"
        f"- Always put code inside a fenced markdown block with a language tag, like {FENCE}python ... {FENCE}.\n"
        "- Never invent facts or code that is not supported by the context.\n\n"
        f"Context:\n{context}"
    )
    return ([{"role": "system", "content": system}]
            + history[-HISTORY_TURNS:]
            + [{"role": "user", "content": question}])


# ───────────────────────── Message view (with code blocks) ─────────────────────────
class MessageView:
    def __init__(self, app, parent, role):
        self.app = app
        self._last_w = 0
        is_user = role == "user"
        bg = C_USER if is_user else C_BOT

        self.outer = ctk.CTkFrame(parent, fg_color="transparent")
        self.outer.pack(fill="x", padx=16, pady=6)

        ctk.CTkLabel(self.outer, text="You" if is_user else BOT_NAME,
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color="#8da2d6" if is_user else "#7ee0b5"
                     ).pack(anchor="e" if is_user else "w", padx=6)

        bubble = ctk.CTkFrame(self.outer, corner_radius=14, fg_color=bg)
        bubble.pack(fill="x", padx=(120, 0) if is_user else (0, 120))

        # The Text sits inside a holder whose height is set in PIXELS (see fit()).
        # Sizing the Text by "number of lines" leaves empty space at the bottom,
        # because code lines / small gap lines are shorter than the base font.
        self.holder = tk.Frame(bubble, bg=bg, height=30)
        self.holder.pack(fill="x", padx=3, pady=3)
        self.holder.pack_propagate(False)

        t = self.text = tk.Text(
            self.holder, wrap="word", bg=bg, fg="#e8eaf0", bd=0, highlightthickness=0,
            relief="flat", font=("Segoe UI", 12), padx=14, pady=PAD_Y, height=1,
            cursor="arrow", insertwidth=0, spacing1=2, spacing3=2)
        t.pack(fill="both", expand=True)

        t.tag_config("bold", font=("Segoe UI", 12, "bold"))
        t.tag_config("inline", font=("Consolas", 11), background="#2b3040", foreground="#ffb86c")
        t.tag_config("codehead", font=("Segoe UI", 9, "bold"), background="#161b22",
                     foreground="#8b949e", lmargin1=12, lmargin2=12, spacing1=8)
        t.tag_config("code", font=("Consolas", 11), background="#0d1117", foreground="#e6edf3",
                     lmargin1=12, lmargin2=12, rmargin=12, wrap="char", spacing1=1, spacing3=1)
        t.tag_config("gap", font=("Segoe UI", 4))

        t.bind("<Configure>", self._on_configure)
        t.bind("<MouseWheel>", self._forward_wheel)

    # --- rendering ---
    def set_text(self, raw: str, final: bool = False):
        t = self.text
        t.configure(state="normal")
        t.delete("1.0", "end")
        pos = 0
        for m in CODE_RE.finditer(raw):
            self._prose(raw[pos:m.start()])
            self._code(m.group(1), m.group(2), final)
            pos = m.end()
        self._prose(raw[pos:])
        if t.get("end-2c", "end-1c") == "\n":
            t.delete("end-2c", "end-1c")
        t.configure(state="disabled")
        self.fit()

    def _prose(self, s: str):
        s = s.strip("\n")
        if not s.strip():
            return
        t = self.text
        for line in s.split("\n"):
            line = line.rstrip()
            heading = re.match(r"^#{1,6}\s+(.*)", line)
            if heading:
                self._inline(heading.group(1), ("bold",))
            else:
                self._inline(re.sub(r"^\s*[*-]\s+", "• ", line), ())
            t.insert("end", "\n")

    def _inline(self, s: str, base: tuple):
        t = self.text
        for piece in INLINE_RE.split(s):
            if not piece:
                continue
            if len(piece) > 2 and piece.startswith("`") and piece.endswith("`"):
                t.insert("end", piece[1:-1], base + ("inline",))
            elif len(piece) > 4 and piece.startswith("**") and piece.endswith("**"):
                t.insert("end", piece[2:-2], base + ("bold",))
            else:
                t.insert("end", piece, base)

    def _code(self, lang: str, body: str, final: bool):
        t = self.text
        body = body.rstrip("\n")
        t.insert("end", f"  {lang or 'code'}  ", ("codehead",))
        if final:
            btn = tk.Button(t, text="Copy", relief="flat", bd=0, padx=8, cursor="hand2",
                            bg="#30363d", fg="#c9d1d9", activebackground="#484f58",
                            activeforeground="#ffffff", font=("Segoe UI", 8),
                            command=lambda b=body: self._copy(b))
            t.window_create("end", window=btn, padx=6)
        t.insert("end", "\n", ("codehead",))
        for ln in body.split("\n"):
            t.insert("end", ln + "\n", ("code",))
        t.insert("end", "\n", ("code",))   # bottom padding of the block
        t.insert("end", "\n", ("gap",))    # gap before the next text

    def _copy(self, s: str):
        self.app.clipboard_clear()
        self.app.clipboard_append(s)

    def add_sources(self, titles):
        if titles:
            ctk.CTkLabel(self.outer, text="Sources: " + "  ·  ".join(titles),
                         font=ctk.CTkFont(size=11), text_color="#6b7390",
                         wraplength=640, justify="left").pack(anchor="w", padx=8, pady=(2, 0))

    # --- sizing ---
    def fit(self):
        t = self.text
        t.update_idletasks()
        n = t.count("1.0", "end", "update", "ypixels")
        px = n[0] if isinstance(n, (tuple, list)) else n
        self.holder.configure(height=max(30, int(px or 0) + 2 * PAD_Y + 2))

    def _on_configure(self, e):
        if e.width != self._last_w:
            self._last_w = e.width
            self.fit()

    def _forward_wheel(self, e):
        self.app.chat._parent_canvas.yview_scroll(-int(e.delta / 6), "units")
        return "break"


# ───────────────────────── App ─────────────────────────
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.title("Eiva - Article Assistant")
        self.geometry("1120x740")
        self.minsize(840, 520)

        self.kb = KnowledgeBase()
        self.events: queue.Queue = queue.Queue()
        self.history: list[dict] = []
        self.buttons: dict[str, ctk.CTkButton] = {}
        self.current: MessageView | None = None
        self.current_raw = ""
        self.last_question = ""
        self.sources: list[str] = []
        self.had_error = False
        self.ready = False

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_chat()
        self._set_busy(True)

        self.after(40, self._poll)
        threading.Thread(target=self._load_kb, daemon=True).start()

    # --- UI ---
    def _build_sidebar(self):
        sb = ctk.CTkFrame(self, width=290, corner_radius=0, fg_color=C_SIDEBAR)
        sb.grid(row=0, column=0, sticky="nsew")
        sb.grid_propagate(False)
        sb.grid_rowconfigure(2, weight=1)
        sb.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(sb, text="📚  Articles", font=ctk.CTkFont(size=20, weight="bold")
                     ).grid(row=0, column=0, sticky="w", padx=20, pady=(22, 2))
        ctk.CTkLabel(sb, text=f"{len(self.kb.files)} articles", text_color="#6b7390",
                     font=ctk.CTkFont(size=12)).grid(row=1, column=0, sticky="w", padx=22, pady=(0, 8))

        lst = ctk.CTkScrollableFrame(sb, fg_color="transparent")
        lst.grid(row=2, column=0, sticky="nsew", padx=6)

        if not self.kb.files:
            ctk.CTkLabel(lst, text="The 'articles' folder is empty\nor was not found.",
                         text_color="#8a8fa3").pack(pady=20)
        for path in self.kb.files:
            title = self.kb.title_of(path)
            short = title if len(title) <= 34 else title[:33] + "…"
            btn = ctk.CTkButton(lst, text=short, anchor="w", height=38, corner_radius=8,
                                fg_color="transparent", hover_color=C_HOVER, text_color="#c9d1e0",
                                command=lambda p=path: self.show_article(p))
            btn.pack(fill="x", pady=1)
            self.buttons[title] = btn

        self.status = ctk.CTkLabel(sb, text="Getting ready…", text_color="#8a8fa3",
                                   font=ctk.CTkFont(size=12))
        self.status.grid(row=3, column=0, sticky="w", padx=20, pady=(8, 2))
        self.progress = ctk.CTkProgressBar(sb, height=6)
        self.progress.set(0)
        self.progress.grid(row=4, column=0, sticky="ew", padx=20, pady=(0, 18))

    def _build_chat(self):
        main = ctk.CTkFrame(self, corner_radius=0, fg_color=C_MAIN)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(1, weight=1)

        head = ctk.CTkFrame(main, height=58, corner_radius=0, fg_color="#12141b")
        head.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(head, text=f"🤖  {BOT_NAME}", font=ctk.CTkFont(size=17, weight="bold")
                     ).pack(side="left", padx=22, pady=14)
        ctk.CTkLabel(head, text=LANGUAGE_MODEL.split("/")[-1], text_color="#5b6280",
                     font=ctk.CTkFont(size=11)).pack(side="right", padx=22)

        self.chat = ctk.CTkScrollableFrame(main, fg_color=C_MAIN)
        self.chat.grid(row=1, column=0, sticky="nsew")
        # one scroll "unit" = SCROLL_STEP pixels; a mouse-wheel notch = 20 units
        self.chat._parent_canvas.configure(yscrollincrement=SCROLL_STEP)

        bar = ctk.CTkFrame(main, fg_color="transparent")
        bar.grid(row=2, column=0, sticky="ew", padx=18, pady=14)
        bar.grid_columnconfigure(0, weight=1)
        self.entry = ctk.CTkEntry(bar, height=46, corner_radius=23, font=ctk.CTkFont(size=14),
                                  placeholder_text="Ask a question about the articles…")
        self.entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.entry.bind("<Return>", lambda e: self.send())
        self.send_btn = ctk.CTkButton(bar, text="Send  ➤", width=100, height=46,
                                      corner_radius=23, command=self.send)
        self.send_btn.grid(row=0, column=1)

    def _set_busy(self, busy: bool):
        state = "disabled" if busy else "normal"
        self.entry.configure(state=state)
        self.send_btn.configure(state=state)
        if not busy:
            self.entry.focus()

    def highlight(self, titles):
        for title, btn in self.buttons.items():
            btn.configure(fg_color=C_ACCENT if title in titles else "transparent")

    def scroll_bottom(self):
        self.update_idletasks()
        self.chat._parent_canvas.yview_moveto(1.0)

    def show_article(self, path: Path):
        try:
            content = self.kb.article_text(path)
        except (OSError, UnicodeError) as e:
            content = f"Could not read the file: {e}"
        win = ctk.CTkToplevel(self)
        win.title(self.kb.title_of(path))
        win.geometry("780x620")
        box = ctk.CTkTextbox(win, wrap="word", font=ctk.CTkFont(size=14))
        box.pack(fill="both", expand=True, padx=12, pady=12)
        box.insert("1.0", content)
        box.configure(state="disabled")
        win.after(150, win.focus)

    # --- logic ---
    def _load_kb(self):
        try:
            self.kb.build(lambda d, t: self.events.put(("progress", d, t)))
            self.events.put(("ready",))
        except Exception as e:  # thread boundary: any error must reach the UI
            self.events.put(("load_error", str(e)))

    def send(self):
        q = self.entry.get().strip()
        if not q or not self.ready:
            return
        self.entry.delete(0, "end")
        self._set_busy(True)
        self.highlight(())
        self.last_question, self.sources, self.had_error = q, [], False

        MessageView(self, self.chat, "user").set_text(q, final=True)
        self.current = MessageView(self, self.chat, "assistant")
        self.current_raw = ""
        self.current.set_text("…")
        self.scroll_bottom()
        threading.Thread(target=self._answer, args=(q, list(self.history)), daemon=True).start()

    def _answer(self, question, history):
        try:
            hits = [h for h in self.kb.retrieve(question) if h[1] >= MIN_SIMILARITY]
            titles = list(dict.fromkeys(c.title for c, _ in hits))
            self.events.put(("sources", titles))
            if not hits:
                self.events.put(("token", NOT_FOUND))
                return
            stream = ollama.chat(model=LANGUAGE_MODEL,
                                 messages=build_messages(question, hits, history),
                                 stream=True, options={"temperature": 0.2})
            for part in stream:
                self.events.put(("token", part["message"]["content"]))
        except Exception as e:  # thread boundary
            self.events.put(("error", str(e)))
        finally:
            self.events.put(("done",))

    def _poll(self):
        dirty = False
        try:
            while True:
                ev = self.events.get_nowait()
                kind = ev[0]
                if kind == "progress":
                    self.progress.set(ev[1] / ev[2])
                    self.status.configure(text=f"Building index… {ev[1]}/{ev[2]}")
                elif kind == "ready":
                    self.ready = True
                    self.progress.grid_remove()
                    self.status.configure(text=f"✓ Ready - {len(self.kb.chunks)} chunks indexed")
                    MessageView(self, self.chat, "assistant").set_text(
                        f"Hi! I'm {BOT_NAME} 👋\nAsk me anything about the articles in the sidebar.",
                        final=True)
                    self._set_busy(False)
                elif kind == "load_error":
                    self.status.configure(text="✗ Error")
                    MessageView(self, self.chat, "assistant").set_text(
                        f"⚠️ I couldn't prepare the articles:\n{ev[1]}\n\n"
                        f"Make sure Ollama is running and the model is installed:\n"
                        f"{FENCE}bash\nollama pull {EMBEDDING_MODEL}\n{FENCE}", final=True)
                elif kind == "sources":
                    self.sources = ev[1]
                    self.highlight(ev[1])
                elif kind == "token":
                    self.current_raw += ev[1]
                    dirty = True
                elif kind == "error":
                    self.had_error = True
                    self.current_raw += f"\n\n⚠️ Error: {ev[1]}"
                    dirty = True
                elif kind == "done":
                    self._finish()
                    dirty = False
        except queue.Empty:
            pass
        if dirty and self.current:
            self.current.set_text(self.current_raw)
            self.scroll_bottom()
        self.after(40, self._poll)

    def _finish(self):
        if not self.current:
            return
        self.current.set_text(self.current_raw or NOT_FOUND, final=True)
        if not self.had_error and self.current_raw:
            self.current.add_sources(self.sources)
            self.history += [{"role": "user", "content": self.last_question},
                             {"role": "assistant", "content": self.current_raw}]
        self.current = None
        self._set_busy(False)
        self.scroll_bottom()


if __name__ == "__main__":
    App().mainloop()