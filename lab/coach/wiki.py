"""Tiny on-demand Grand Piece Online wiki retriever.

This is an ENGINEERED knowledge helper for rare/current game details that are
too large or too changeable to bake into the runtime prompt.  It uses the
public MediaWiki API, caches aggressively, strips markup, and never blocks the
biological brain/control threads because it is called only by the low-rate
semantic coach worker.
"""

from __future__ import annotations

import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path


API = "https://grand-piece-online.fandom.com/api.php"
UA = "DigitalFlyLab/1.0 semantic-coach research"


def _clean(text: str) -> str:
    text = html.unescape(str(text or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class GPOWikiRetriever:
    def __init__(self, cache_path: Path, timeout_s: float = 6.0):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.timeout_s = float(timeout_s)
        try:
            self.cache = json.loads(
                self.cache_path.read_text(encoding="utf-8"))
            if not isinstance(self.cache, dict):
                self.cache = {}
        except Exception:
            self.cache = {}

    def _get(self, params: dict) -> dict:
        url = API + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(
            url, headers={"User-Agent": UA, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def lookup(self, query: str, max_chars: int = 6500) -> str:
        query = _clean(query)[:120]
        if not query:
            return ""
        key = query.lower()
        cached = self.cache.get(key)
        if isinstance(cached, dict):
            age = time.time() - float(cached.get("wall_s", 0.0))
            if age < 86400.0 and cached.get("text"):
                return str(cached["text"])[:max_chars]

        search = self._get({
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 3,
            "utf8": 1,
            "format": "json",
            "origin": "*",
        })
        hits = ((search.get("query") or {}).get("search") or [])[:3]
        titles = [str(h.get("title") or "") for h in hits if h.get("title")]
        if not titles:
            return ""

        pages = self._get({
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "exsectionformat": "plain",
            "redirects": 1,
            "titles": "|".join(titles),
            "format": "json",
            "origin": "*",
        })
        records = []
        for page in ((pages.get("query") or {}).get("pages") or {}).values():
            title = _clean(page.get("title"))
            extract = _clean(page.get("extract"))
            if not extract:
                continue
            records.append(f"## {title}\n{extract[:2800]}")

        text = "\n\n".join(records)[:max_chars]
        if text:
            self.cache[key] = {
                "wall_s": time.time(),
                "query": query,
                "titles": titles,
                "text": text,
            }
            tmp = self.cache_path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(self.cache, indent=1, ensure_ascii=False),
                encoding="utf-8")
            tmp.replace(self.cache_path)
        return text
