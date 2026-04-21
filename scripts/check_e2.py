"""Verifica artefactos minimos del entregable E2 (rutas del curso + estructura nueva)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def exists(rel: str) -> bool:
    return (ROOT / rel).exists()


def any_exists(paths: list[str]) -> bool:
    return any(exists(p) for p in paths)


def diagrams_ok() -> bool:
    """Acepta diagramas en docs/architecture (legado) o docs/diagrams (nueva estructura)."""
    legacy_c4 = any_exists(
        [
            "docs/architecture/architecture_general_es.svg",
            "docs/architecture/architecture_general_es.png",
        ]
    )
    legacy_flow = any_exists(
        [
            "docs/architecture/data_flow_es.svg",
            "docs/architecture/data_flow_es.png",
        ]
    )
    if legacy_c4 and legacy_flow:
        return True

    c4_context = any_exists(
        [
            "docs/diagrams/c4-context.svg",
            "docs/diagrams/c4-context.png",
        ]
    )
    c4_container = any_exists(
        [
            "docs/diagrams/c4-container.svg",
            "docs/diagrams/c4-container.png",
        ]
    )
    data_flow = any_exists(
        [
            "docs/diagrams/data-flow.svg",
            "docs/diagrams/data-flow.png",
        ]
    )
    return c4_context and c4_container and data_flow


def main() -> int:
    missing: list[str] = []

    if not diagrams_ok():
        missing.append(
            "diagramas: use docs/architecture/*_es.svg|png "
            "o docs/diagrams/c4-context|png, c4-container|png, data-flow|png"
        )

    required_files = [
        "docs/adr/ADR-001.md",
        "docs/adr/ADR-002.md",
        "docs/adr/ADR-003.md",
        "docs/adr/ADR-001-openai-as-llm-base.md",
        "docs/adr/ADR-002-postgresql-pgvector-as-vector-store.md",
        "docs/adr/ADR-003-fastapi-as-framework.md",
        "docs/api/openapi.yaml",
        "docs/PROJECT_DOCUMENTATION.md",
    ]
    for rel in required_files:
        if not exists(rel):
            missing.append(rel)

    if missing:
        print("FALTAN E2:")
        for item in missing:
            print(f"  - {item}")
        return 1

    print("OK: E2 completo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
