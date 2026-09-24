"""Extração de texto de arquivos anexados no chat (PDF, TXT, MD, CSV...).

Os anexos chegam como lista de dicts: {"filename": str, "mime": str, "content_base64": str}.
O texto extraído é injetado como contexto no prompt (semelhante ao RAG).
Limites conservadores para não estourar a janela de contexto do modelo local.
"""

import base64
import io
import logging
import os

logger = logging.getLogger(__name__)

MAX_FILES = 6
MAX_CHARS_PER_FILE = 6000
MAX_TOTAL_CHARS = 12000

TEXT_EXTENSIONS = {'.txt', '.md', '.markdown', '.csv', '.json', '.log'}


def _decode_text(raw: bytes) -> str:
    for encoding in ('utf-8', 'latin-1'):
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, ValueError):
            continue
    return raw.decode('utf-8', errors='replace')


def _extract_pdf(raw: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return '[PDF não lido: biblioteca pypdf ausente no servidor]'
    try:
        reader = PdfReader(io.BytesIO(raw))
        pages = []
        for page in reader.pages[:20]:
            try:
                pages.append(page.extract_text() or '')
            except Exception:
                continue
        return '\n'.join(pages)
    except Exception as e:
        logger.warning('Falha ao extrair PDF: %s', e)
        return '[PDF não pôde ser lido]'


def _extract_docx(raw: bytes) -> str:
    try:
        import docx
    except ImportError:
        return '[DOCX não lido: instale python-docx no servidor]'
    try:
        doc = docx.Document(io.BytesIO(raw))
        return '\n'.join(p.text for p in doc.paragraphs)
    except Exception as e:
        logger.warning('Falha ao extrair DOCX: %s', e)
        return '[DOCX não pôde ser lido]'


def extract_attachments_text(attachments) -> str:
    """Recebe a lista de anexos e devolve o contexto pronto p/ o prompt ('' se vazio)."""
    if not attachments:
        return ''

    blocks = []
    total = 0
    for att in (attachments or [])[:MAX_FILES]:
        if not isinstance(att, dict):
            continue
        filename = (att.get('filename') or 'arquivo').strip() or 'arquivo'
        try:
            raw = base64.b64decode(att.get('content_base64') or '', validate=False)
        except Exception:
            blocks.append(f'--- {filename} ---\n[arquivo inválido]')
            continue
        if not raw or len(raw) > 10 * 1024 * 1024:
            blocks.append(f'--- {filename} ---\n[arquivo vazio ou maior que 10MB]')
            continue

        ext = os.path.splitext(filename.lower())[1]
        if ext == '.pdf':
            text = _extract_pdf(raw)
        elif ext == '.docx':
            text = _extract_docx(raw)
        elif ext in TEXT_EXTENSIONS or not ext:
            text = _decode_text(raw)
        else:
            text = f'[tipo {ext or "desconhecido"} sem extração de texto suportada]'

        text = ' '.join((text or '').split())
        if len(text) > MAX_CHARS_PER_FILE:
            text = text[:MAX_CHARS_PER_FILE] + '… [trecho cortado]'
        if total + len(text) > MAX_TOTAL_CHARS:
            text = text[:max(0, MAX_TOTAL_CHARS - total)] + '… [trecho cortado]'
        total += len(text)
        blocks.append(f'--- {filename} ---\n{text}')
        if total >= MAX_TOTAL_CHARS:
            break

    if not blocks:
        return ''
    return 'ARQUIVOS ANEXADOS PELO USUÁRIO:\n' + '\n\n'.join(blocks)
