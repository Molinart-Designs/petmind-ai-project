"""Consultas rápidas a la BD (equivalente al python -c largo). Ej.: docker compose exec api python scripts/db_smoke.py"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from sqlalchemy import text
from src.db.session import engine

conn = engine.connect()
try:
    print(
        conn.execute(
            text("SELECT extname FROM pg_extension WHERE extname='vector'")
        ).fetchall()
    )
    print(
        conn.execute(
            text("SELECT tablename FROM pg_tables WHERE tablename='document_chunks'")
        ).fetchall()
    )
finally:
    conn.close()
