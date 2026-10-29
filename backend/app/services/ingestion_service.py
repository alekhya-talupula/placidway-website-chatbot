
import asyncio
import hashlib
import ipaddress
import socket
import uuid

from pathlib import Path
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser

import chromadb
import yaml
import httpx
from bs4 import BeautifulSoup

from app.config import settings
from app.services.embedding_service import embed_texts

UA = "PlacidWayAssessmentBot/1.0 (polite knowledge ingestion)"
_last_fetch = 0.0
_lock = asyncio.Lock()


def read_sources():
    sources_path = Path(settings.sources_file)

    with sources_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or not isinstance(data.get("sources"), list):
        raise ValueError("sources.yaml must contain a 'sources' list.")

    sources = data["sources"]

    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ValueError(f"Source {index + 1} must be an object.")

        if not source.get("id"):
            raise ValueError(f"Source {index + 1} is missing 'id'.")

        if not source.get("title"):
            raise ValueError(f"Source {index + 1} is missing 'title'.")

        if not source.get("url"):
            raise ValueError(f"Source {index + 1} is missing 'url'.")

    return sources


def validate_url(url: str):
    p = urlparse(url)

    if p.scheme != "https" or not p.hostname or p.username or p.password:
        raise ValueError("Only credential-free HTTPS URLs are accepted.")

    host = p.hostname.lower()

    if host == "localhost" or host.endswith(".local"):
        raise ValueError("Local destinations are blocked.")

    try:
        addresses = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise ValueError(f"Host could not be resolved: {host}") from e

    for addr in addresses:
        ip = ipaddress.ip_address(addr[4][0])
        if not ip.is_global:
            raise ValueError("Non-public IP destinations are blocked.")

    return host


async def polite_wait():
    global _last_fetch

    async with _lock:
        loop = asyncio.get_running_loop()
        now = loop.time()
        delay = max(0, 1.0 - (now - _last_fetch))

        if delay:
            await asyncio.sleep(delay)

        _last_fetch = loop.time()


async def fetch_approved(url: str) -> tuple[str, str]:
    host = validate_url(url)
    robots_url = f"https://{host}/robots.txt"

    async with httpx.AsyncClient(
        timeout=20,
        follow_redirects=False,
        headers={"User-Agent": UA},
    ) as client:
        await polite_wait()
        robots_resp = await client.get(robots_url)

        if robots_resp.status_code == 200:
            rp = RobotFileParser()
            rp.parse(robots_resp.text.splitlines())

            if not rp.can_fetch(UA, url) or not rp.can_fetch("*", url):
                raise ValueError("robots.txt disallows this page.")

        elif robots_resp.status_code not in (404, 410):
            raise ValueError(
                f"Could not verify robots.txt (HTTP {robots_resp.status_code})."
            )

        current = url

        for _ in range(4):
            current_host = validate_url(current)

            if current_host != host:
                raise ValueError(
                    "Cross-host redirects are blocked; verify approved URL."
                )

            await polite_wait()
            response = await client.get(current)

            if response.status_code in (301, 302, 303, 307, 308):
                target = urljoin(
                    current, response.headers.get("location", "")
                )

                if urlparse(target).hostname != host:
                    raise ValueError(
                        "Cross-host redirects are blocked; verify approved URL."
                    )

                current = target
                continue

            response.raise_for_status()
            ctype = response.headers.get("content-type", "")

            if "text/html" not in ctype:
                raise ValueError("Source is not an HTML page.")

            return response.text, current

    raise ValueError("Too many redirects.")


def clean_html(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else "Untitled page"

    for el in soup(
        ["script", "style", "noscript", "svg", "nav", "footer", "header"]
    ):
        el.decompose()

    main = soup.find("main") or soup.find("article") or soup.body or soup

    return title, "\n".join(
        x.strip() for x in main.stripped_strings if x.strip()
    )


def chunks(text: str, size=1100, overlap=160):
    text = " ".join(text.split())

    if not text:
        return []

    result = []
    start = 0

    while start < len(text):
        end = min(len(text), start + size)

        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + size // 2:
                end = boundary

        result.append(text[start:end])

        if end == len(text):
            break

        start = max(start + 1, end - overlap)

    return result


async def ingest_all():
    sources = read_sources()
    configured = [s for s in sources if s.get("url", "").strip()]

    if not configured:
        raise ValueError("Add exact approved URLs to app/sources.yaml first.")

    # Fetch and process all pages before changing the active collection.
    records = []

    for source in configured:
        html, final_url = await fetch_approved(source["url"].strip())
        _, text = clean_html(html)
        pieces = chunks(text)

        if not pieces:
            raise ValueError(
                f"No readable content extracted from {source['title']}"
            )

        for i, piece in enumerate(pieces):
            digest = hashlib.sha256(
                (source["id"] + piece).encode()
            ).hexdigest()

            records.append({
                "id": digest,
                "text": piece,
                "source_id": source["id"],
                "title": source["title"],
                "url": final_url,
                "chunk_index": i,
            })

    # Generate embeddings in smaller batches to reduce peak memory usage.
    vectors = []
    embedding_batch_size = 32

    for i in range(0, len(records), embedding_batch_size):
        batch_texts = [
            record["text"]
            for record in records[i:i + embedding_batch_size]
        ]
        batch_vectors = embed_texts(batch_texts)
        vectors.extend(batch_vectors)

    # Store the complete dataset in a new collection.
    persist_dir = Path(settings.chroma_persist_directory)
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(persist_dir)
    )

    generation = "pw_" + uuid.uuid4().hex

    col = client.create_collection(
        generation,
        metadata={"hnsw:space": "cosine"},
    )

    storage_batch_size = 128

    for i in range(0, len(records), storage_batch_size):
        part = records[i:i + storage_batch_size]

        col.add(
            ids=[record["id"] for record in part],
            documents=[record["text"] for record in part],
            embeddings=vectors[i:i + storage_batch_size],
            metadatas=[
                {
                    key: value
                    for key, value in record.items()
                    if key not in ("id", "text")
                }
                for record in part
            ],
        )

    # Update the active collection pointer only after successful storage.
    pointer = persist_dir / "active_collection.txt"
    temporary_pointer = pointer.with_suffix(".tmp")

    temporary_pointer.write_text(generation, encoding="utf-8")
    temporary_pointer.replace(pointer)

    return {
        "source_pages": len(configured),
        "chunks": len(records),
        "collection": generation,
    }
