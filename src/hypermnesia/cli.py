"""`hypermnesia` console entry point.

With no arguments (or `serve`) it runs the MCP server, so the Docker CMD and
existing setups keep working. The admin subcommands (export / import /
reindex) run on the server host with direct DB access — see admin.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .admin import export_memories, import_memories, reindex
from .config import get_settings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hypermnesia",
        description="Semantic memory store for AI agents (MCP server + admin CLI).",
    )
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("serve", help="run the MCP server (the default)")

    p = sub.add_parser(
        "export",
        help="dump all memories (archived included, embeddings excluded) as JSONL",
    )
    p.add_argument("--out", default="-", help="output file, or - for stdout (default)")
    p.add_argument(
        "--scope", action="append", dest="scopes", metavar="SCOPE",
        help="limit to a scope; repeatable (default: everything)",
    )

    p = sub.add_parser(
        "import",
        help="restore a JSONL dump, re-embedding with the configured model; "
        "existing ids are skipped, never overwritten",
    )
    p.add_argument("file", help="JSONL dump produced by `hypermnesia export`")
    p.add_argument("--batch-size", type=int, default=64)

    p = sub.add_parser(
        "reindex",
        help="re-embed every memory with the configured HM_EMBEDDING_* model "
        "(the migration path for swapping models); restart the server after",
    )
    p.add_argument("--batch-size", type=int, default=64)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    cmd = args.cmd or "serve"

    if cmd == "serve":
        from .server import main as serve

        serve()
        return

    settings = get_settings()
    try:
        if cmd == "export":
            if args.out == "-":
                count = asyncio.run(
                    export_memories(settings.database_url, sys.stdout, args.scopes)
                )
            else:
                with open(args.out, "w", encoding="utf-8") as f:
                    count = asyncio.run(
                        export_memories(settings.database_url, f, args.scopes)
                    )
            print(f"exported {count} memories", file=sys.stderr)
        elif cmd == "import":
            result = asyncio.run(
                import_memories(settings, args.file, batch_size=args.batch_size)
            )
            print(json.dumps(result))
        elif cmd == "reindex":
            result = asyncio.run(reindex(settings, batch_size=args.batch_size))
            print(json.dumps(result))
            print(
                "reindexed — restart the server with the new HM_EMBEDDING_* settings",
                file=sys.stderr,
            )
    except (RuntimeError, ValueError, OSError) as e:
        raise SystemExit(f"error: {e}") from e


if __name__ == "__main__":
    main()
