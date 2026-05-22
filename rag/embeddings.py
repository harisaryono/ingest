import time
import requests
from typing import List

from config import OLLAMA_BASE, EMBED_MODEL, EMBED_BATCH_SIZE, EMBED_RETRY_COUNT, EMBED_TIMEOUT

EMBED_URL = f"{OLLAMA_BASE}/api/embed"


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    all_embeddings = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        for attempt in range(EMBED_RETRY_COUNT):
            try:
                resp = requests.post(
                    EMBED_URL,
                    json={"model": EMBED_MODEL, "input": batch},
                    timeout=EMBED_TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()
                all_embeddings.extend(data["embeddings"])
                break
            except Exception as e:
                if attempt == EMBED_RETRY_COUNT - 1:
                    raise RuntimeError(
                        f"Embedding failed after {EMBED_RETRY_COUNT} attempts "
                        f"for batch {i}: {e}"
                    )
                time.sleep(2**attempt)
    return all_embeddings
