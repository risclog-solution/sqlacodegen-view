from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Any, Callable, TypedDict

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
    parse_aggregate_row,
    parse_extension_row,
    parse_function_row,
    parse_policy_row,
    parse_publication_row,
    parse_trigger_row,
)
from sqlacodegen.seed_export import export_pgdata_py

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

    # ----------- Mapping für Schleife vorbereiten ------------
    class ExportDict(TypedDict, total=False):
        title: str
        only_outfile: bool
        file: str
        file2: str
        gen_func: Callable[[Any], tuple[Any, Any]]
        entities_varname: str
        template: str
        statement: str
        parse_row_func: Callable[..., Any]

    EXPORTS = [
        {
            "title": "Tabellen-Models",
            "only_outfile": True,
            "gen_func": lambda generator: generator.generate(),
            "file": "orm_tables.py",
        },
        {
            "title": "Views-Models",
            "only_outfile": False,
            "gen_func": lambda generator: generator.generate(),
            "file": "orm_views.py",
            "file2": "pg_views.py",
        },
        {
            "title": "Funktionen",
            "entities_varname": "all_functions",
            "template": "ALEMBIC_FUNCTION_TEMPLATE",
            "statement": "ALEMBIC_FUNCTION_STATEMENT",
            "parse_row_func": parse_function_row,
            "file": "pg_functions.py",
        },
        {
            "title": "Policies",
            "entities_varname": "all_policies",
            "template": "ALEMBIC_POLICIES_TEMPLATE",
            "statement": "ALEMBIC_POLICIES_STATEMENT",
            "parse_row_func": parse_policy_row,
            "file": "pg_policies.py",
        },
        {
            "title": "Triggers",
            "entities_varname": "all_triggers",
            "template": "ALEMBIC_TRIGGER_TEMPLATE",
            "statement": "ALEMBIC_TRIGGER_STATEMENT",
            "parse_row_func": parse_trigger_row,
            "file": "pg_triggers.py",
        },
        {
            "title": "Aggregates",
            "entities_varname": "all_aggregates",
            "template": "ALEMBIC_AGGREGATE_TEMPLATE",
            "statement": "ALEMBIC_AGGREGATE_STATEMENT",
            "parse_row_func": parse_aggregate_row,
            "file": "pg_aggregates.py",
        },
        {
            "title": "Extensions",
            "entities_varname": "all_extensions",
            "template": "ALEMBIC_EXTENSION_TEMPLATE",
            "statement": "ALEMBIC_EXTENSION_STATEMENT",
            "parse_row_func": parse_extension_row,
            "file": "pg_extensions.py",
        },
        {
            "title": "Publications",
            "entities_varname": "all_publications",
            "template": "ALEMBIC_PUBLICATION_TEMPLATE",
            "statement": "ALEMBIC_PUBLICATION_STATEMENT",
            "parse_row_func": parse_publication_row,
            "file": "pg_publications.py",
        },
    ]

    # ----------- Export-Loop ------------
    for export in EXPORTS:
        title = str(export["title"])
        gen_func = export.get("gen_func")

        if "only_outfile" in export and export["only_outfile"]:
            if args.outfile_dir:
                dest_path = Path(str(parent), str(export["file"]))
                generator_tables = generator_class(metadata_tables, engine, options)
                orm_tables, _ = gen_func(generator_tables)  # type: ignore[operator]
                with open(dest_path, "w", encoding="utf-8") as f:
                    f.write(orm_tables)
                print(f"{title} geschrieben nach: {dest_path.as_posix()}")
            else:
                generator_tables = generator_class(metadata_tables, engine, options)
                orm_tables, _ = gen_func(generator_tables)  # type: ignore[operator]
                print(f"### {title.upper()} ###")
                print(orm_tables)
            continue

        if title == "Views-Models":
            if args.outfile_dir:
                dest_orm_path = Path(str(parent), str(export["file"]))
                dest_pg_path = Path(str(parent), str(export["file2"]))
                generator_views = generator_class(metadata_views, engine, options)
                orm_views, pg_alembic = gen_func(generator_views)  # type: ignore[operator]
                with open(dest_orm_path, "w", encoding="utf-8") as f:
                    f.write(orm_views)
                if pg_alembic:    
                    with open(dest_pg_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(pg_alembic))
                print(f"{title} geschrieben nach: {dest_orm_path.as_posix()}")
            else:
                generator_views = generator_class(metadata_views, engine, options)
                orm_views, pg_alembic = gen_func(generator_views)  # type: ignore[operator]
                print(f"### {title.upper()} ###")
                print(orm_views)
            continue

        # Alles andere (alembic_utils)
        entities_varname = export["entities_varname"]
        template = export["template"]
        statement = export["statement"]
        parse_row_func = export["parse_row_func"]
        file_name = export["file"]

        if args.outfile_dir:
            if title == "Sequences":
                generator_functions = generator_tables.generate_alembic_utils_sequences(
                    template=template,
                    statement=statement,
                    parse_row_func=parse_row_func,
                    schema=args.schemas or "public",
                    entities_varname=entities_varname,
                )
            else:
                generator_functions = generator_tables.generate_alembic_utils_entities(
                    template=template,
                    statement=statement,
                    parse_row_func=parse_row_func,
                    schema=args.schemas or "public",
                    entities_varname=entities_varname,
                )
            dest_pg_path = Path(str(parent), str(file_name))
            with open(dest_pg_path, "w", encoding="utf-8") as f:
                f.write("\n".join(generator_functions))
            print(f"{title} geschrieben nach: {dest_pg_path.as_posix()}")
        else:
            if title == "Sequences":
                generator_functions = generator_tables.generate_alembic_utils_sequences(
                    template=template,
                    statement=statement,
                    parse_row_func=parse_row_func,
                    schema=args.schemas or "public",
                    entities_varname=entities_varname,
                )
            else:
                generator_functions = generator_tables.generate_alembic_utils_entities(
                    template=template,
                    statement=statement,
                    parse_row_func=parse_row_func,
                    schema=args.schemas or "public",
                    entities_varname=entities_varname,
                )
            print(f"### {title.upper()} ###")
            print(generator_functions)

    # ----------- PGData SEED Export separat ------------
    if args.outfile_dir:
        all_view_names = set()
        for schema in schemas:
            all_view_names |= set(inspector.get_view_names(schema=schema))

        dest_pg_path = Path(str(parent), "pg_seeds.py")
        export_pgdata_py(
            engine=engine,
            metadata=metadata_tables,
            out_path=dest_pg_path,
            view_table_names=all_view_names,
        )
        print(f"PGData Seed geschrieben nach: {dest_pg_path.as_posix()}")

        # ----------- Factories & Fixtures Export separat ------------
        def get_all_models(base: type[Any]) -> list[type[Any]]:
            seen = set()
            todo = list(base.__subclasses__())
            models = []
            while todo:
                m = todo.pop()
                if m not in seen and hasattr(m, "__tablename__"):
                    models.append(m)
                    seen.add(m)
                    todo.extend(m.__subclasses__())
            return models

        def make_dynamic_models(metadata: MetaData) -> dict[str, type[Any]]:
            from sqlalchemy.ext.declarative import declarative_base

            Base = declarative_base(metadata=metadata)
            models_by_table = {}
            for table in metadata.tables.values():
                class_name = "".join([w.capitalize() for w in table.name.split("_")])
                model = type(class_name, (Base,), {"__table__": table})
                models_by_table[table.name] = model
            return models_by_table
