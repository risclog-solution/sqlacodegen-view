from __future__ import annotations

import datetime
import decimal
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import MetaData, select
from sqlalchemy.engine import Engine


def python_literal_value(val: Any, _imports: set[str]) -> str:
    if isinstance(val, str):
        return repr(val)
    if isinstance(val, bool):
        return "True" if val else "False"
    if val is None:
        return "None"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        return repr(val)
    if isinstance(val, decimal.Decimal):
        _imports.add("import decimal")
        return f"decimal.Decimal({str(val)!r})"
    if isinstance(val, datetime.datetime):
        _imports.add("import datetime")
        return f"datetime.datetime.fromisoformat({val.isoformat()!r})"
    if isinstance(val, datetime.date):
        _imports.add("import datetime")
        return f"datetime.date.fromisoformat({val.isoformat()!r})"
    if isinstance(val, datetime.time):
        _imports.add("import datetime")
        return f"datetime.time.fromisoformat({val.isoformat()!r})"
    if isinstance(val, uuid.UUID):
        _imports.add("import uuid")
        return f"uuid.UUID({str(val)!r})"
    if isinstance(val, list):
        return (
            "[" + ", ".join(python_literal_value(item, _imports) for item in val) + "]"
        )
    if isinstance(val, dict):
        items = ", ".join(
            f"{python_literal_value(k, _imports)}: {python_literal_value(v, _imports)}"
            for k, v in val.items()
        )
        return "{" + items + "}"
    raise TypeError(f"Unsupported type for seed export: {type(val)}")


def data_as_code(data: dict[str, list[dict[str, Any]]]) -> tuple[str, set[str]]:
    imports: set[str] = set()
    blocks: list[str] = []
    for tname, rows in data.items():
        rowblocks: list[str] = []
        for row in rows:
            items = ", ".join(
                f"{repr(k)}: {python_literal_value(v, imports)}" for k, v in row.items()
            )
            rowblocks.append(f"        {{{items}}},")
        blocks.append(f"    {repr(tname)}: [\n" + "\n".join(rowblocks) + "\n    ],")
    return "\n".join(blocks), imports


def get_table_dependency_order(metadata: MetaData) -> list[str]:
    from collections import defaultdict

    graph: dict[str, set[str]] = defaultdict(set)
    for table in metadata.tables.values():
        graph[table.name]

    for table in metadata.tables.values():
        name = table.name
        for fk in table.foreign_keys:
            parent = fk.column.table.name
            if parent != name:
                graph[name].add(parent)

    try:
        from graphlib import TopologicalSorter

        ts = TopologicalSorter(graph)
        order = list(ts.static_order())
    except ImportError:
        visited: set[str] = set()
        result: list[str] = []

        def visit(node: str) -> None:
            if node in visited:
                return
            visited.add(node)
            for dep in graph[node]:
                visit(dep)
            result.append(node)

        for node in graph:
            visit(node)
        order = result[::-1]

    return order


def export_pgdata_py(
    engine: Engine, metadata: MetaData, out_path: Path, max_rows: int | None = None
) -> None:
    order = get_table_dependency_order(metadata)
    data: dict[str, list[dict[str, Any]]] = {}

    with engine.connect() as conn:
        for name in order:
            if name not in metadata.tables:
                continue
            table = metadata.tables[name]
            stmt = select(table)
            if max_rows is not None:
                stmt = stmt.limit(max_rows)
            rows = conn.execute(stmt).fetchall()
            result: list[dict[str, Any]] = []
            for row in rows:
                d = {col: getattr(row, col) for col in table.columns.keys()}
                result.append(d)
            data[name] = result

    seed_block, imports = data_as_code(data)
    lines: list[str] = []
    for imp in sorted(imports):
        lines.append(imp)
    lines.append("\n\nall_seeds = {\n" + seed_block + "\n}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
