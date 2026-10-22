from processing import DocumentProcessor

RAW_PDF = """ACME S.A. — Manual interno — Pág. 1
Política de devoluciones■■
Los clientes pueden devolver un producto dentro de los 30 días    corridos desde la compra, siempre que conserven el ticket.
ACME S.A. — Manual interno — Pág. 2
El reintegro se hace por el mismo medio de pago.
"""


def test_clean_text_drops_headers_junk_and_repeated_spaces():
    cleaned = DocumentProcessor().clean_text(RAW_PDF)
    assert "Pág." not in cleaned
    assert "■" not in cleaned
    assert "  " not in cleaned
    assert "Política de devoluciones" in cleaned
    assert "\n" in cleaned


def test_chunks_stay_within_the_token_budget_and_overlap():
    processor = DocumentProcessor(chunk_size=40, chunk_overlap=10)
    text = " ".join(f"politica-{index}" for index in range(300))
    chunks = processor.process_document(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert processor.calculate_tokens(chunk) <= 40
    # Token overlap does not always land on a word boundary, but the tail
    # of one chunk is still repeated at the start of the next.
    previous = chunks[0].split()
    following = chunks[1].split()
    assert previous[-2:] == following[:2]


def test_default_chunk_settings_meet_the_spec():
    processor = DocumentProcessor()
    assert processor.chunk_size >= 500
    assert processor.chunk_overlap == 50
