from __future__ import annotations

import argparse
import json
from pathlib import Path

from nare3.dsm.memory import DynamicSegmentedMemory, sparse_attention_cost


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="nare3", description="Dynamic Segmented Memory CLI")
    parser.add_argument("--store", default=".nare3/dsm.json", help="DSM JSON storage path")
    sub = parser.add_subparsers(dest="command", required=True)

    write = sub.add_parser("write", help="Write text into segmented memory")
    write.add_argument("text", nargs="?", default="")
    write.add_argument("--file")
    write.add_argument("--category")
    write.add_argument("--description")
    write.add_argument("--importance", type=float, default=0.5)

    route = sub.add_parser("route", help="Route a query to relevant memory segments")
    route.add_argument("query")
    route.add_argument("-k", type=int, default=5)
    route.add_argument("--json", action="store_true")

    context = sub.add_parser("context", help="Build active context for a query")
    context.add_argument("query")
    context.add_argument("-k", type=int, default=5)
    context.add_argument("--budget", type=int, default=100_000)

    sub.add_parser("stats", help="Show memory statistics")
    sub.add_parser("rebuild", help="Rebuild category hierarchy and graph links")

    cost = sub.add_parser("cost", help="Compare dense vs DSM active attention cost")
    cost.add_argument("--total", type=int, required=True)
    cost.add_argument("--active", type=int, required=True)

    args = parser.parse_args(argv)
    memory = DynamicSegmentedMemory(storage_path=args.store)

    if args.command == "write":
        text = Path(args.file).read_text(encoding="utf-8") if args.file else args.text
        segments = memory.write(
            text,
            category_path=args.category,
            description=args.description,
            importance=args.importance,
        )
        memory.save()
        print(f"written_segments={len(segments)}")
        for segment in segments:
            print(f"{segment.id} {' → '.join(segment.category_path)} {segment.description}")
    elif args.command == "route":
        results = memory.route(args.query, k=args.k)
        if args.json:
            print(
                json.dumps(
                    [
                        {
                            "id": item.segment.id,
                            "category": list(item.segment.category_path),
                            "description": item.segment.description,
                            "similarity": item.similarity,
                            "priority_score": item.priority_score,
                            "total_score": item.total_score,
                            "graph_distance": item.graph_distance,
                        }
                        for item in results
                    ],
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            for item in results:
                print(
                    f"{item.total_score:.3f} sim={item.similarity:.3f} "
                    f"prio={item.priority_score:.3f} "
                    f"{item.segment.id} {' → '.join(item.segment.category_path)}"
                )
                print(f"  {item.segment.description}")
        memory.save()
    elif args.command == "context":
        active = memory.active_context(args.query, k=args.k, token_budget=args.budget)
        print(active.context_text)
        print(f"\n[estimated_tokens={active.estimated_tokens}/{active.token_budget}]")
        memory.save()
    elif args.command == "stats":
        print(json.dumps(memory.stats(), ensure_ascii=False, indent=2))
    elif args.command == "rebuild":
        moved = memory.rebuild_structure()
        memory.save()
        print(f"moved_segments={moved}")
    elif args.command == "cost":
        print(json.dumps(sparse_attention_cost(args.active, args.total), indent=2))
