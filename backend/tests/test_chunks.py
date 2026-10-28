from app.services.ingestion_service import chunks, clean_html

def test_chunks_retain_text():
    text = "Package price is USD 1200. " * 200
    parts = chunks(text, size=300, overlap=30)
    assert len(parts) > 1
    assert all(parts)

def test_clean_html_removes_scripts():
    title, text = clean_html("<html><head><title>Example</title></head><body><script>bad()</script><main>Hello evidence</main></body></html>")
    assert title == "Example"
    assert "Hello evidence" in text
    assert "bad()" not in text
