import os
import pypandoc
import tempfile
import re
from functools import lru_cache
import requests

LUA_REMOVE_LINKS_IMAGES = """
function Link(el)  return el.content end
function Image(el) return {} end
"""


def remove_macro_calls(text: str, macros: set[str]) -> str:
    n = len(text)
    i = 0
    out = []

    openers = {"[": "]", "(": ")", "{": "}"}

    def skip_spaces(pos: int) -> int:
        while pos < n and text[pos].isspace():
            pos += 1
        return pos

    def consume_balanced(pos: int) -> int | None:
        if text[pos] not in openers:
            return pos
        open_ch = text[pos]
        close_ch = openers[open_ch]
        depth = 0
        j = pos
        while j < n:
            ch = text[j]
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return j + 1
            j += 1
        return None

    while i < n:
        if text[i] != "\\":
            out.append(text[i])
            i += 1
            continue

        j = i + 1
        if j >= n or not text[j].isalpha():
            out.append(text[i])
            i += 1
            continue

        k = j
        while k < n and text[k].isalpha():
            k += 1
        name = text[j:k]

        if name not in macros:
            out.append(text[i])
            i += 1
            continue

        p = skip_spaces(k)
        if p < n:
            end = consume_balanced(p)
            if end is not None:
                i = end
                continue

        i += 1

    return "".join(out)


API_UA = os.getenv("API_UA")
API_TOKEN = os.getenv("WIKI_TOKEN")


def _rest_headers():
    if not API_TOKEN or not API_UA:
        raise RuntimeError(
            "Set WIKI_TOKEN and API_UA to use api.wikimedia.org endpoints."
        )
    return {
        "Authorization": f"Bearer {API_TOKEN}",
        "User-Agent": API_UA,
    }


def html_to_latex(html: str) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        f1 = os.path.join(tmpdir, "rm_links_images.lua")
        with open(f1, "w", encoding="utf-8") as fp:
            fp.write(LUA_REMOVE_LINKS_IMAGES)
        extra = [
            f"--lua-filter={f1}",
            "--mathml",
            "--wrap=none",
        ]
        UNWANTED = {
            "textsuperscript",
            "textsubscript",
            "protect",
            "phantomsection",
            "label",
            "href",
            "url",
            "includegraphics",
            "pandocbounded",
            "displaystyle",
        }

        latex = pypandoc.convert_text(html, to="latex", format="html", extra_args=extra)
        latex = remove_macro_calls(latex, UNWANTED)
        return latex


@lru_cache(maxsize=512)
def search_resources(query: str, limit: int = 10, lang: str = "en"):
    url = f"https://api.wikimedia.org/core/v1/wikipedia/{lang}/search/page"
    params = {"q": query, "limit": str(limit)}
    r = requests.get(url, params=params, headers=_rest_headers())
    r.raise_for_status()
    data = r.json()
    pages = data.get("pages", [])
    out = []
    for p in pages:
        out.append(
            {
                "title": p.get("title"),
                "page_id": p.get("key"),
                "description": p.get("description") or p.get("excerpt") or "",
            }
        )
    return out


@lru_cache(maxsize=512)
def get_page_text(title: str, lang: str = "en") -> str:
    url = f"https://api.wikimedia.org/core/v1/wikipedia/{lang}/page/{title}/with_html"
    r = requests.get(url, headers=_rest_headers())
    r.raise_for_status()
    data = r.json()

    return data


_SECTION_LEVELS = [
    ("part", -1),
    ("chapter", 0),
    ("section", 1),
    ("subsection", 2),
    ("subsubsection", 3),
    ("paragraph", 4),
    ("subparagraph", 5),
]


_CMD_RE = re.compile(
    r"\\(?P<cmd>part|chapter|section|subsection|subsubsection|paragraph|subparagraph)\*?"
    r"(?:\s*\[[^\[\]]*\])?\s*\{",
)


def _extract_brace_group(s: str, start: int):
    assert s[start] == "{"
    i = start + 1
    depth = 1
    n = len(s)
    content_chars = []
    while i < n and depth > 0:
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                i += 1
                break
        if depth > 0:
            content_chars.append(ch)
        i += 1
    return "".join(content_chars).strip(), i


def extract_outline_from_latex(tex: str):
    level_map = {name: lvl for name, lvl in _SECTION_LEVELS}

    items = []
    pos = 0
    n = len(tex)

    while True:
        m = _CMD_RE.search(tex, pos)
        if not m:
            break
        cmd = m.group("cmd")
        starred = bool(tex[m.start() : m.start() + len(cmd) + 2].endswith("*"))
        brace_idx = tex.find("{", m.end() - 1)
        if brace_idx == -1:
            pos = m.end()
            continue
        title, after = _extract_brace_group(tex, brace_idx)
        items.append(
            {
                "title": title,
                "level": level_map.get(cmd, 6),
                "cmd": cmd,
                "starred": starred,
                "children": [],
            }
        )
        pos = after

    roots = []
    stack = []

    for node in items:
        while stack and stack[-1]["level"] >= node["level"]:
            stack.pop()
        if stack:
            stack[-1]["children"].append(node)
        else:
            roots.append(node)
        stack.append(node)

    return roots


def pretty_outline_from_latex(tex: str) -> str:
    tree = extract_outline_from_latex(tex)  # from the earlier step [1][2]

    display = {
        "part": "Part",
        "chapter": "Chapter",
        "section": "Section",
        "subsection": "Subsection",
        "subsubsection": "Subsubsection",
        "paragraph": "Paragraph",
        "subparagraph": "Subparagraph",
    }

    lines = []
    node_id = 0

    def walk(nodes, indent=0):
        nonlocal node_id
        for n in nodes:
            node_id += 1
            label = display.get(n["cmd"], n["cmd"].capitalize())
            pad = "    " * indent
            title = n["title"] or ""
            lines.append(f"{pad}{label} (id={node_id}) {title}")
            if n["children"]:
                walk(n["children"], indent + 1)

    walk(tree, 0)
    return "\n".join(lines)


def index_outline_with_spans(tex: str):
    level_map = {name: lvl for name, lvl in _SECTION_LEVELS}
    items = []

    pos = 0
    while True:
        m = _CMD_RE.search(tex, pos)
        if not m:
            break
        cmd = m.group("cmd")
        brace_idx = tex.find("{", m.end() - 1)
        if brace_idx == -1:
            pos = m.end()
            continue
        title, after = _extract_brace_group(tex, brace_idx)
        items.append(
            {
                "cmd": cmd,
                "level": level_map.get(cmd, 6),
                "title": title,
                "start": m.start(),
                "end_title": after,
            }
        )
        pos = after
    return items


def truncate_latex_to_id(tex: str, target_id: int):
    target_id = int(target_id)
    items = index_outline_with_spans(tex)
    if not items:
        return ""

    if target_id < 1 or target_id > len(items):
        return ""

    node = items[target_id - 1]
    start = node["start"]
    level = node["level"]

    end = len(tex)
    for nxt in items[target_id:]:
        if nxt["level"] <= level:
            end = nxt["start"]
            break

    return tex[start:end]
