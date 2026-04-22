import sys
from pathlib import Path

# Permite `python scripts/check_db.py` sin que sys.path sea solo `scripts/`
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from sqlalchemy import text
from src.db.session import engine

with engine.connect() as conn:
    vector_ext = conn.execute(
        text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
    ).fetchall()
    chunks_table = conn.execute(
        text("SELECT tablename FROM pg_tables WHERE tablename = 'document_chunks'")
    ).fetchall()

print("pgvector extension:", vector_ext)
print("document_chunks table:", chunks_table)
