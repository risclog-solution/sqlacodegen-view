from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import inspect
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import MetaData

try:
    import citext
except ImportError:
    citext = None

try:
    import geoalchemy2
except ImportError:
    geoalchemy2 = None

try:
    import pgvector.sqlalchemy
except ImportError:
    pgvector = None

from sqlacodegen.risclog_generators import (
    parse_function_row,
    parse_policy_row,
    parse_trigger_row,
)

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points, version
else:
    from importlib.metadata import entry_points, version


def _parse_engine_arg(arg_str: str) -> tuple[str, Any]:
    if "=" not in arg_str:
        raise argparse.ArgumentTypeError("engine-arg must be in key=value format")

    key, value = arg_str.split("=", 1)
    try:
        value = ast.literal_eval(value)
    except Exception:
        pass  # Leave as string if literal_eval fails

    return key, value


def _parse_engine_args(arg_list: list[str]) -> dict[str, Any]:
    result = {}
    for arg in arg_list or []:
        key, value = _parse_engine_arg(arg)
        result[key] = value

    return result


def main() -> None:
    generators = {ep.name: ep for ep in entry_points(group="sqlacodegen.generators")}
    parser = argparse.ArgumentParser(
        description="Generates SQLAlchemy model code from an existing database."
    )
    parser.add_argument("url", nargs="?", help="SQLAlchemy url to the database")
    parser.add_argument(
        "--options", help="options (comma-delimited) passed to the generator class"
    )
    parser.add_argument(
        "--version", action="store_true", help="print the version number and exit"
    )
    parser.add_argument(
        "--schemas", help="load tables from the given schemas (comma-delimited)"
    )
    parser.add_argument(
        "--generator",
        choices=generators,
        default="declarative",
        help="generator class to use",
    )
    parser.add_argument(
        "--tables", help="tables to process (comma-delimited, default: all)"
    )
    parser.add_argument(
        "--noviews",
        action="store_true",
        help="ignore views (always true for sqlmodels generator)",
    )
    parser.add_argument(
        "--engine-arg",
        action="append",
        help=(
            "engine arguments in key=value format, e.g., "
            '--engine-arg=connect_args=\'{"user": "scott"}\' '
            "--engine-arg thick_mode=true or "
            '--engine-arg thick_mode=\'{"lib_dir": "/path"}\' '
            "(values are parsed with ast.literal_eval)"
        ),
    )
    parser.add_argument("--outfile", help="file to write output to (default: stdout)")

    parser.add_argument(
        "--outfile-dir", help="directory to write generated model files to (optional)"
    )
    args = parser.parse_args()

    if args.version:
        print(version("sqlacodegen"))
        return

    if not args.url:
        print("You must supply a url\n", file=sys.stderr)
        parser.print_help()
        return

    if citext:
        print(f"Using sqlalchemy-citext {version('sqlalchemy-citext')}")

    if geoalchemy2:
        print(f"Using geoalchemy2 {version('geoalchemy2')}")

    if pgvector:
        print(f"Using pgvector {version('pgvector')}")

    # Use reflection to fill in the metadata
    engine_args = _parse_engine_args(args.engine_arg)
    engine = create_engine(args.url, **engine_args)
    metadata = MetaData()
    tables = args.tables.split(",") if args.tables else None
    schemas = args.schemas.split(",") if args.schemas else [None]
    options = set(args.options.split(",")) if args.options else set()

    # Instantiate the generator
    generator_class = generators[args.generator].load()
    generator = generator_class(metadata, engine, options)

    if not generator.views_supported:
        name = generator_class.__name__
        print(
            f"VIEW models will not be generated when using the '{name}' generator",
            file=sys.stderr,
        )

    for schema in schemas:
        metadata.reflect(
            engine, schema, (generator.views_supported and not args.noviews), tables
        )

    inspector = inspect(engine)
    all_view_names = set()
    for schema in schemas:
        all_view_names |= set(inspector.get_view_names(schema=schema))

    table_names = []
    view_names = []
    for table in metadata.tables.values():
        name = table.name
        if name in all_view_names:
            view_names.append(name)
        else:
            table_names.append(name)

    # Separate MetaData reflektieren
    metadata_tables = MetaData()
    for schema in schemas:
        metadata_tables.reflect(engine, schema=schema, only=table_names, views=False)

    metadata_views = MetaData()
    for schema in schemas:
        metadata_views.reflect(engine, schema=schema, only=view_names, views=True)

    if args.outfile_dir:
        parent = Path(args.outfile_dir)
        parent.mkdir(parents=True, exist_ok=True)

    # Tabellen-Models
    if args.outfile_dir:
        dest_orm_path = Path(parent, "orm_tables.py")
        with open(dest_orm_path, "w", encoding="utf-8") as f:
            generator_tables = generator_class(metadata_tables, engine, options)
            orm_tables, _ = generator_tables.generate()
            f.write(orm_tables)
        print(f"Tabellen-Models geschrieben nach: {dest_orm_path.as_posix()}")
    else:
        generator_tables = generator_class(metadata_tables, engine, options)
        orm_tables, _ = generator_tables.generate()
        print("### TABELLEN-MODELLE ###")
        print(orm_tables)

    # Views-Models
    if args.outfile_dir:
        dest_orm_path = Path(parent, "orm_views.py")
        dest_pg_path = Path(parent, "pg_views.py")
        generator_views = generator_class(metadata_views, engine, options)
        orm_views, pg_alembic = generator_views.generate()
        with open(dest_orm_path, "w", encoding="utf-8") as f:
            f.write(orm_views)
        with open(dest_pg_path, "w", encoding="utf-8") as f:
            f.write("\n".join(pg_alembic))
        print(f"View-Models geschrieben nach: {dest_orm_path.as_posix()}")
    else:
        generator_views = generator_class(metadata_views, engine, options)
        orm_views, pg_alembic = generator_views.generate()
        print("### VIEW-MODELLE ###")
        print(orm_views)

    # Funktionen
    if args.outfile_dir:
        generator_functions = generator_tables.generate_alembic_utils_entities(
            template="ALEMBIC_FUNCTION_TEMPLATE",
            statement="ALEMBIC_FUNCTION_STATEMENT",
            parse_row_func=parse_function_row,
            schema=args.schemas or "public",
            entities_varname="all_functions",
        )
        dest_pg_path = Path(parent, "pg_functions.py")
        with open(dest_pg_path, "w", encoding="utf-8") as f:
            f.write("\n".join(generator_functions))

        print(f"Funktionen geschrieben nach: {dest_pg_path.as_posix()}")
    else:
        generator_functions = generator_tables.generate_alembic_utils_entities(
            template="ALEMBIC_FUNCTION_TEMPLATE",
            statement="ALEMBIC_FUNCTION_STATEMENT",
            parse_row_func=parse_function_row,
            schema=args.schemas or "public",
            entities_varname="all_functions",
        )
        print("### FUNKTIONEN ###")
        print(generator_functions)

    # Policies
    if args.outfile_dir:
        generator_functions = generator_tables.generate_alembic_utils_entities(
            template="ALEMBIC_POLICIES_TEMPLATE",
            statement="ALEMBIC_POLICIES_STATEMENT",
            parse_row_func=parse_policy_row,
            schema=args.schemas or "public",
            entities_varname="all_policies",
        )
        dest_pg_path = Path(parent, "pg_policies.py")
        with open(dest_pg_path, "w", encoding="utf-8") as f:
            f.write("\n".join(generator_functions))

        print(f"Policies geschrieben nach: {dest_pg_path.as_posix()}")
    else:
        generator_functions = generator_tables.generate_alembic_utils_entities(
            template="ALEMBIC_POLICIES_TEMPLATE",
            statement="ALEMBIC_POLICIES_STATEMENT",
            parse_row_func=parse_policy_row,
            schema=args.schemas or "public",
            entities_varname="all_policies",
        )
        print("### Policies ###")
        print(generator_functions)

    # # Triggers
    if args.outfile_dir:
        generator_functions = generator_tables.generate_alembic_utils_entities(
            template="ALEMBIC_TRIGGER_TEMPLATE",
            statement="ALEMBIC_TRIGGER_STATEMENT",
            parse_row_func=parse_trigger_row,
            schema=args.schemas or "public",
            entities_varname="all_triggers",
        )
        dest_pg_path = Path(parent, "pg_triggers.py")
        with open(dest_pg_path, "w", encoding="utf-8") as f:
            f.write("\n".join(generator_functions))

        print(f"Triggers geschrieben nach: {dest_pg_path.as_posix()}")
    else:
        generator_functions = generator_tables.generate_alembic_utils_entities(
            template="ALEMBIC_TRIGGER_TEMPLATE",
            statement="ALEMBIC_TRIGGER_STATEMENT",
            parse_row_func=parse_trigger_row,
            schema=args.schemas or "public",
            entities_varname="all_triggers",
        )
        print("### Triggers ###")
        print(generator_functions)
