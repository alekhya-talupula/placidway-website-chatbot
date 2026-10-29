
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

    if not isinstance(data, dict) or not isinstance(
        data.get("sources"), list
    ):
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
    parsed = urlparse(url)

    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError(
            "Only credential-free HTTPS URLs are accepted."
        )

    host = parsed.hostname.lower()

    if host == "localhost" or host.endswith(".local"):
        raise ValueError("Local destinations are blocked.")

    try:
        addresses = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ValueError(
            f"Host could not be resolved: {host}"
        ) from exc

    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])

        if not ip.is_global:
            raise ValueError(
                "Non-public IP destinations are blocked."
            )

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
        robots_response = await client.get(robots_url)

        if robots_response.status_code == 200:
            parser = RobotFileParser()
            parser.parse(robots_response.text.splitlines())

            if (
                not parser.can_fetch(UA, url)
                or not parser.can_fetch("*", url)
            ):
                raise ValueError(
                    "robots.txt disallows this page."
                )

        elif robots_response.status_code not in (404, 410):
            raise ValueError(
                "Could not verify robots.txt "
                f"(HTTP {robots_response.status_code})."
            )

        current = url

        for _ in range(4):
            current_host = validate_url(current)

            if current_host != host:
                raise ValueError(
                    "Cross-host redirects are blocked; "
                    "verify the approved URL."
                )

            await polite_wait()
            response = await client.get(current)

            if response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("location", "")

                if not location:
                    raise ValueError(
                        "Redirect response has no Location header."
                    )

                target = urljoin(current, location)

                if urlparse(target).hostname != host:
                    raise ValueError(
                        "Cross-host redirects are blocked; "
                        "verify the approved URL."
                    )

                current = target
                continue

            # A 403 or another HTTP error is handled by the caller,
            # which records the error and continues to the next source.
            response.raise_for_status()

            content_type = response.headers.get(
                "content-type", ""
            )

            if "text/html" not in content_type.lower():
                raise ValueError("Source is not an HTML page.")

            return response.text, current

    raise ValueError("Too many redirects.")


def clean_html(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")

    title = (
        soup.title.get_text(" ", strip=True)
        if soup.title
        else "Untitled page"
    )

    for element in soup(
        [
            "script",
            "style",
            "noscript",
            "svg",
            "nav",
            "footer",
            "header",
        ]
    ):
        element.decompose()

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.body
        or soup
    )

    text = "\n".join(
        item.strip()
        for item in main.stripped_strings
        if item.strip()
    )

    return title, text


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

    configured = [
        source
        for source in sources
        if source.get("url", "").strip()
    ]

    if not configured:
        raise ValueError(
            "Add exact approved URLs to app/sources.yaml first."
        )

    # Fetch each source independently. A failed page must not
    # automatically prevent other approved pages from being indexed.
    records = []
    source_errors = []
    successful_pages = 0

    for source in configured:
        try:
            html, final_url = await fetch_approved(
                source["url"].strip()
            )

            _, page_text = clean_html(html)
            page_chunks = chunks(page_text)

            if not page_chunks:
                source_errors.append({
                    "source": source["title"],
                    "error": "No readable content extracted.",
                })
                continue

            for index, piece in enumerate(page_chunks):
                digest = hashlib.sha256(
                    (
                        source["id"] + piece
                    ).encode("utf-8")
                ).hexdigest()

                records.append({
                    "id": digest,
                    "text": piece,
                    "source_id": source["id"],
                    "title": source["title"],
                    "url": final_url,
                    "chunk_index": index,
                })

            successful_pages += 1

        except Exception as exc:
            # Keep the error for diagnostics without stopping
            # ingestion of the remaining sources.
            source_errors.append({
                "source": source["title"],
                "error": str(exc),
            })

    # Never replace the active index with an empty collection.
    if not records:
        raise ValueError(
            "No approved source pages could be indexed. "
            f"Source errors: {source_errors}"
        )

    # Generate embeddings in smaller batches to reduce peak memory.
    vectors = []
    embedding_batch_size = 32

    for start in range(0, len(records), embedding_batch_size):
        batch = records[
            start:start + embedding_batch_size
        ]

        batch_texts = [
            record["text"] for record in batch
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

    collection = client.create_collection(
        generation,
        metadata={"hnsw:space": "cosine"},
    )

    storage_batch_size = 128

    try:
        for start in range(0, len(records), storage_batch_size):
            batch = records[
                start:start + storage_batch_size
            ]

            batch_vectors = vectors[
                start:start + storage_batch_size
            ]

            collection.add(
                ids=[
                    record["id"] for record in batch
                ],
                documents=[
                    record["text"] for record in batch
                ],
                embeddings=batch_vectors,
                metadatas=[
                    {
                        key: value
                        for key, value in record.items()
                        if key not in ("id", "text")
                    }
                    for record in batch
                ],
            )

        # Switch the active index only after successful storage.
        pointer = persist_dir / "active_collection.txt"
        temporary_pointer = persist_dir / "active_collection.tmp"

        temporary_pointer.write_text(
            generation,
            encoding="utf-8",
        )

        temporary_pointer.replace(pointer)

    except Exception:
        # Clean up the incomplete new collection if storage fails.
        try:
            client.delete_collection(generation)
        except Exception:
            pass

        raise

    return {
        "source_pages_configured": len(configured),
        "source_pages_indexed": successful_pages,
        "source_pages_failed": len(source_errors),
        "chunks": len(records),
        "collection": generation,
        "source_errors": source_errors,
    }
