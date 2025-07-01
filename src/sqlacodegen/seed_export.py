from __future__ import annotations

import datetime
import decimal
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import MetaData, select, text
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
        name = table.name
        for fk in table.foreign_keys:
            parent = fk.column.table.name
            if parent != name:
                graph[name].add(parent)

    visited: set[str] = set()
    result: list[str] = []

    def visit(node: str) -> None:
        if node in visited:
            return
        visited.add(node)
        for dep in graph[node]:
            visit(dep)
        result.append(node)

    for table in metadata.tables.values():
        visit(table.name)
    return result[::-1]


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

        sequence_rows = conn.execute(
            text("""
            SELECT sequence_schema, sequence_name
            FROM information_schema.sequences
            WHERE sequence_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY sequence_schema, sequence_name
            """)
        ).fetchall()
        raw_sql_stmts = []
        for row in sequence_rows:
            schema = row.sequence_schema
            seq_name = row.sequence_name
            lastval = conn.execute(
                text(f"SELECT last_value FROM {schema}.{seq_name}")
            ).scalar()
            raw_sql_stmts.append(
                f"        SELECT setval('{schema}.{seq_name}', {lastval}, false);"
            )
        raw_sql_str = "\n".join(raw_sql_stmts)

    seed_block, imports = data_as_code(data)
    lines: list[str] = []
    for imp in sorted(imports):
        lines.append(imp)
    lines.append("\n\nall_seeds = {\n" + seed_block + "\n}")

    lines.append('\nall_seeds[\'sql_next_values\'] = """\n' + raw_sql_str + '\n"""\n')

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
