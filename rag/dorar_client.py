import json
import os
import re
from functools import lru_cache
from typing import Dict, List

import requests
from bs4 import BeautifulSoup

from config import DORAR_API_URL

WEBSHARE_PROXY_LIST_URL = "https://proxy.webshare.io/api/v2/proxy/list/"
WEBSHARE_ENV_FALLBACKS = [
    os.getenv("WEBSHARE_API_KEY_FILE", ""),
    "/media/harry/DATA120B/GIT/youtube_transcript_bundle/.env.local",
]
DORAR_ALLOW_DIRECT = os.getenv("DORAR_ALLOW_DIRECT", "").strip().lower() in {"1", "true", "yes", "on"}


def _parse_json_or_jsonp(text: str):
    raw = (text or "").strip()
    if not raw:
        return {}
    if raw.startswith("{") or raw.startswith("["):
        return json.loads(raw)

    match = re.match(r"^[^(]+\((.*)\)\s*;?\s*$", raw, re.S)
    if match:
        return json.loads(match.group(1))
    return json.loads(raw)


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html or "", "lxml")
    text = soup.get_text("\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_candidate(item: Dict) -> Dict:
    raw_html = item.get("th") or item.get("text") or item.get("hadith") or ""
    text = clean_html(raw_html)
    return {
        "text": text,
        "raw_html": raw_html,
        "source": item.get("sourcename") or item.get("source") or item.get("book") or "",
        "grade": item.get("hukm") or item.get("grade") or item.get("verdict") or "",
        "author": item.get("muhaddith") or item.get("author") or "",
        "raw": item,
    }


def _load_webshare_api_key() -> str:
    key = (os.getenv("WEBSHARE_API_KEY") or "").strip()
    if key:
        return key
    for path in WEBSHARE_ENV_FALLBACKS:
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    name, value = line.split("=", 1)
                    if name.strip() == "WEBSHARE_API_KEY":
                        return value.strip().strip('"').strip("'")
        except Exception:
            continue
    return ""


@lru_cache(maxsize=1)
def _load_webshare_proxy_urls() -> List[str]:
    key = _load_webshare_api_key()
    if not key:
        return []
    try:
        resp = requests.get(
            WEBSHARE_PROXY_LIST_URL,
            headers={"Authorization": f"Token {key}"},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    results = data.get("results") if isinstance(data, dict) else []
    proxies: List[str] = []
    for item in results or []:
        if not isinstance(item, dict):
            continue
        host = item.get("proxy_address") or item.get("host") or item.get("proxy_host") or item.get("address")
        port = item.get("port")
        username = item.get("username") or item.get("user") or item.get("login")
        password = item.get("password") or item.get("pass") or item.get("secret")
        if host and port and username and password:
            proxies.append(f"http://{username}:{password}@{host}:{port}")
    return proxies


def _proxy_cycle(request_key: str = "") -> List[str]:
    proxies = _load_webshare_proxy_urls()
    if not proxies:
        return []
    if not request_key:
        return proxies
    seed = sum(ord(ch) for ch in request_key) % len(proxies)
    return proxies[seed:] + proxies[:seed]


def _extract_candidates(data: Dict) -> List[Dict]:
    items = data.get("ahadith") or data.get("ahadiths") or data.get("results") or []
    candidates: List[Dict] = []
    for item in items:
        if isinstance(item, dict):
            candidates.append(_normalize_candidate(item))
        else:
            candidates.append({
                "text": clean_html(str(item)),
                "raw_html": str(item),
                "source": "",
                "grade": "",
                "author": "",
                "raw": item,
            })
    return candidates


def search_dorar_hadith(query: str, limit: int = 5, timeout: int = 20) -> List[Dict]:
    request_key = query.strip()
    session_headers = {
        "User-Agent": "Mozilla/5.0 (compatible; rag-review-workspace/1.0)",
        "Referer": "https://dorar.net/article/389",
        "Accept": "application/json,text/javascript,*/*;q=0.8",
    }
    proxy_urls = _proxy_cycle(request_key=request_key)
    if not proxy_urls and not DORAR_ALLOW_DIRECT:
        return []

    attempts = ([None] if DORAR_ALLOW_DIRECT else []) + proxy_urls
    for proxy_url in attempts:
        try:
            proxies = None
            if proxy_url:
                proxies = {"http": proxy_url, "https": proxy_url}
            resp = requests.get(
                DORAR_API_URL,
                params={"skey": query, "callback": "?"},
                timeout=(min(5, timeout), min(10, timeout)),
                headers=session_headers,
                proxies=proxies,
            )
            if resp.status_code >= 400:
                continue
            data = _parse_json_or_jsonp(resp.text)
            candidates = _extract_candidates(data)
            if candidates:
                return candidates[:limit]
        except Exception:
            continue
    return []
