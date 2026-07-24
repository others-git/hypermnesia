"""Unit tests for the admin CLI: JSONL serialization + argument parsing.

No DB or embeddings — the full export/import/reindex flows are covered by
tests/test_cli_e2e.py inside the compose stack.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from hypermnesia.admin import EXPORT_FIELDS, line_to_row, row_to_line
from hypermnesia.cli import _build_parser


def _row(**overrides) -> dict:
    base = {
        "id": "3f0e9c1a-7c39-4c1e-9e6b-2a1f5f9d0b11",
        "owner_id": "dev",
        "scope": "project:alpha-1234",
        "type": "fact",
        "content": "naïve café — unicode survives the dump",
        "description": "unicode round-trip",
        "tags": ["a", "b"],
        "metadata": {"nested": [1, 2, {"k": "v"}]},
        "importance": 1.5,
        "created_at": datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 2, 3, 4, 5, 6, tzinfo=timezone.utc),
        "last_accessed_at": datetime(2026, 3, 4, 5, 6, 7, tzinfo=timezone.utc),
        "archived_at": None,
    }
    base.update(overrides)
    return base


def test_jsonl_round_trip_preserves_everything():
    row = _row()
    assert line_to_row(row_to_line(row)) == row


def test_archived_timestamp_round_trips():
    row = _row(archived_at=datetime(2026, 6, 1, tzinfo=timezone.utc))
    back = line_to_row(row_to_line(row))
    assert back["archived_at"] == row["archived_at"]


def test_line_missing_fields_is_rejected():
    with pytest.raises(ValueError, match="missing fields"):
        line_to_row('{"id": "x"}')


def test_export_fields_exclude_embedding():
    # Dumps must survive an embedding-model change, so vectors never land in them.
    assert "embedding" not in EXPORT_FIELDS
    assert "model_id" not in EXPORT_FIELDS


def test_cli_parser_dispatch():
    p = _build_parser()
    assert p.parse_args([]).cmd is None  # defaults to serve
    assert p.parse_args(["serve"]).cmd == "serve"
    args = p.parse_args(["export", "--scope", "a", "--scope", "b"])
    assert args.cmd == "export" and args.scopes == ["a", "b"] and args.out == "-"
    args = p.parse_args(["import", "dump.jsonl"])
    assert args.cmd == "import" and args.file == "dump.jsonl" and args.batch_size == 64
    assert p.parse_args(["reindex"]).batch_size == 64
