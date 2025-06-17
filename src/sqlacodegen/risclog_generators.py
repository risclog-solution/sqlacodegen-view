from typing import TYPE_CHECKING, Any

from sqlalchemy import Column, Index, text, types
from sqlalchemy import types as satypes
from sqlalchemy.engine import Engine

from sqlacodegen.generators import (
    Base,
    DeclarativeGenerator,
    LiteralImport,
    TablesGenerator,
)
from sqlacodegen.models import Model, ModelClass
from sqlacodegen.utils import render_callable

if TYPE_CHECKING:
    from sqlacodegen.generators import TablesGenerator


def sa_type_from_column(col: Column[Any]) -> str:
    t = col.type

    if isinstance(t, (satypes.BIGINT, satypes.BigInteger)):
        return "BigInteger"
    if isinstance(t, (satypes.BINARY, satypes.VARBINARY)):
        return "LargeBinary"
    if isinstance(t, (satypes.BLOB,)):
        return "BLOB"
    if isinstance(t, (satypes.BOOLEAN, satypes.Boolean)):
        return "Boolean"
    if isinstance(t, (satypes.CHAR, satypes.NCHAR)):
        return "String"
    if isinstance(t, (satypes.CLOB,)):
        return "CLOB"
    if isinstance(t, (satypes.DATE, satypes.Date)):
        return "Date"
    if isinstance(t, (satypes.DATETIME, satypes.DateTime, satypes.TIMESTAMP)):
        return "DateTime"
    if isinstance(t, (satypes.DECIMAL, satypes.NUMERIC, satypes.Numeric)):
        return "Numeric"
    if isinstance(t, (satypes.DOUBLE, satypes.Double, satypes.DOUBLE_PRECISION)):
        return "Float"
    if isinstance(t, (satypes.Enum,)):
        enum_class = getattr(t, "enum_class", None)
        if enum_class is not None:
            enum_name = (
                enum_class.__name__
                if hasattr(enum_class, "__name__")
                else str(enum_class)
            )
            return f"Enum({enum_name})"
        if hasattr(t, "enums"):
            return f"Enum({', '.join(repr(e) for e in t.enums)})"
        return "Enum"

    if isinstance(t, (satypes.FLOAT, satypes.Float, satypes.REAL)):
        return "Float"
    if isinstance(
        t,
        (
            satypes.INT,
            satypes.INTEGER,
            satypes.Integer,
            satypes.SMALLINT,
            satypes.SmallInteger,
        ),
    ):
        return "Integer"
    if isinstance(t, (satypes.Interval,)):
        return "Interval"
    if isinstance(t, (satypes.JSON,)):
        return "JSON"
    if isinstance(t, (satypes.LargeBinary,)):
        return "LargeBinary"
    if isinstance(
        t,
        (
            satypes.NVARCHAR,
            satypes.VARCHAR,
            satypes.String,
            satypes.Text,
            satypes.TEXT,
            satypes.Unicode,
            satypes.UnicodeText,
        ),
    ):
        return "String"
    if isinstance(t, (satypes.PickleType,)):
        return "PickleType"
    if isinstance(t, (satypes.Time, satypes.TIME)):
        return "Time"
    if isinstance(t, (satypes.TupleType,)):
        return "TupleType"
    if isinstance(t, (satypes.TypeDecorator,)):
        return "TypeDecorator"
    if isinstance(t, (satypes.UUID, satypes.Uuid)):
        return "Uuid"
    if isinstance(t, (satypes.ARRAY,)):
        item_type = getattr(t, "item_type", None)
        if item_type:
            if isinstance(
                item_type,
                (satypes.Integer, satypes.INT, satypes.SMALLINT, satypes.BigInteger),
            ):
                return "ARRAY(Integer)"
            if isinstance(item_type, (satypes.String, satypes.Text, satypes.VARCHAR)):
                return "ARRAY(String)"
            if isinstance(
                item_type,
                (
                    satypes.Numeric,
                    satypes.DECIMAL,
                    satypes.Float,
                    satypes.DOUBLE,
                    satypes.REAL,
                ),
            ):
                return "ARRAY(Numeric)"
            if isinstance(item_type, (satypes.Boolean,)):
                return "ARRAY(Boolean)"
        return "ARRAY(String)"

    t_str = str(type(t)).upper() + " " + str(t).upper()
    mapping = {
        "BIGINT": "BigInteger",
        "BINARY": "LargeBinary",
        "BLOB": "BLOB",
        "BOOLEAN": "Boolean",
        "CHAR": "String",
        "CLOB": "CLOB",
        "DATE": "Date",
        "DATETIME": "DateTime",
        "DECIMAL": "Numeric",
        "DOUBLE": "Float",
        "ENUM": "Enum",
        "FLOAT": "Float",
        "INT": "Integer",
        "INTEGER": "Integer",
        "INTERVAL": "Interval",
        "JSON": "JSON",
        "LARGEBINARY": "LargeBinary",
        "NCHAR": "String",
        "NUMERIC": "Numeric",
        "NVARCHAR": "String",
        "PICKLETYPE": "PickleType",
        "REAL": "Float",
        "SMALLINT": "Integer",
        "STRING": "String",
        "TEXT": "String",
        "TIME": "Time",
        "TIMESTAMP": "DateTime",
        "TUPLETYPE": "TupleType",
        "TYPEDECORATOR": "TypeDecorator",
        "UNICODE": "String",
        "UNICODETEXT": "String",
        "UUID": "Uuid",
        "VARBINARY": "LargeBinary",
        "VARCHAR": "String",
        "ARRAY": "ARRAY(String)",
    }
    for key, value in mapping.items():
        if key in t_str:
            return value

    return "String"


EXCLUDED_TABLES = {"tmp_functest"}


BASE_META_DATA = Base(
    literal_imports=[
        LiteralImport(
            "risclog.claimxdb.clx.clx_models_base",
            "PortalObject",
        )
    ],
    declarations=[],
    metadata_ref="PortalObject.metadata",
)


def clx_generate_base(self: "TablesGenerator") -> None:
    self.base = BASE_META_DATA


TablesGenerator.generate_base = clx_generate_base  # type: ignore[method-assign]


def get_expression_indexes(
    engine: Engine, table_name: str, schema: str = "public"
) -> list[dict[str, Any]]:
    query = """
    SELECT
        t.relname AS table_name,
        i.relname AS index_name,
        a.amname  AS index_type,
        pg_get_indexdef(ix.indexrelid) AS index_def,
        ix.indisunique AS is_unique
    FROM pg_class t
    JOIN pg_index ix ON t.oid = ix.indrelid
    JOIN pg_class i ON i.oid = ix.indexrelid
    JOIN pg_am a ON i.relam = a.oid
    JOIN pg_namespace n ON n.oid = t.relnamespace
    WHERE t.relkind = 'r'
      AND n.nspname = :schema
      AND t.relname = :table_name
      AND NOT ix.indisprimary
      AND NOT ix.indisexclusion
      AND ix.indisvalid
    """
    with engine.connect() as conn:
        result = conn.execute(text(query), {"table_name": table_name, "schema": schema})
        return [dict(row) for row in result]


def clx_render_index(self: "TablesGenerator", index: Index) -> str:
    elements = []
    opclass_map = {}

    if index.columns:
        for col in index.columns:
            elements.append(repr(col.name))

            if (
                "postgresql" in index.dialect_options
                and index.dialect_options["postgresql"].get("using") == "gin"
                and hasattr(col, "type")
            ):
                coltype = getattr(col.type, "python_type", None)
                if isinstance(col.type, (types.String, types.Text, types.Unicode)) or (
                    coltype and coltype is str
                ):
                    opclass_map[col.name] = "gin_trgm_ops"

    elif getattr(index, "expressions", None):
        for expr in index.expressions:
            expr_str = str(expr).strip()
            elements.append(f"text({expr_str!r})")

            if (
                "postgresql" in index.dialect_options
                and index.dialect_options["postgresql"].get("using") == "gin"
            ):
                if (
                    "::tsvector" not in expr_str
                    and "array" not in expr_str.lower()
                    and "json" not in expr_str.lower()
                ):
                    opclass_map[expr_str] = "gin_trgm_ops"

    if not elements:
        print(
            f"# WARNING: Skipped index {getattr(index, 'name', None)!r} on table {getattr(index.table, 'name', None)!r} (no columns or expressions)."
        )
        return ""

    kwargs: dict[str, Any] = {}

    if index.unique:
        kwargs["unique"] = True

    if "postgresql" in index.dialect_options:
        dialect_opts = index.dialect_options["postgresql"]
        if "using" in dialect_opts:
            using = dialect_opts["using"]
            kwargs["postgresql_using"] = (
                f"'{using}'" if isinstance(using, str) else using
            )

        if opclass_map:
            kwargs["postgresql_ops"] = opclass_map

    return render_callable("Index", repr(index.name), *elements, kwargs=kwargs)


TablesGenerator.render_index = clx_render_index  # type: ignore[method-assign]


class DeclarativeGeneratorWithViews(DeclarativeGenerator):
    def generate_base(self) -> None:
        self.base = BASE_META_DATA

    def render_models(self, models: list[Model]) -> str:
        rendered: list[str] = []

        used_types: set[str] = set()
        type_imports = {
            "text": ("sqlalchemy", "text"),
            "Table": ("sqlalchemy", "Table"),
            "Column": ("sqlalchemy", "Column"),
            "BigInteger": ("sqlalchemy", "BigInteger"),
            "LargeBinary": ("sqlalchemy", "LargeBinary"),
            "BLOB": ("sqlalchemy", "BLOB"),
            "Boolean": ("sqlalchemy", "Boolean"),
            "String": ("sqlalchemy", "String"),
            "CLOB": ("sqlalchemy", "CLOB"),
            "Date": ("sqlalchemy", "Date"),
            "DateTime": ("sqlalchemy", "DateTime"),
            "Numeric": ("sqlalchemy", "Numeric"),
            "Float": ("sqlalchemy", "Float"),
            "Enum": ("sqlalchemy", "Enum"),
            "Integer": ("sqlalchemy", "Integer"),
            "Interval": ("sqlalchemy", "Interval"),
            "JSON": ("sqlalchemy", "JSON"),
            "PickleType": ("sqlalchemy", "PickleType"),
            "Time": ("sqlalchemy", "Time"),
            "TupleType": ("sqlalchemy", "TupleType"),
            "TypeDecorator": ("sqlalchemy", "TypeDecorator"),
            "ARRAY": ("sqlalchemy", "ARRAY"),
        }

        string_types = [
            "CHAR",
            "NCHAR",
            "NVARCHAR",
            "UNICODE",
            "UNICODETEXT",
            "TEXT",
            "VARCHAR",
        ]
        for t in string_types:
            type_imports[t] = ("sqlalchemy", "String")

        largebinary_types = ["BINARY", "VARBINARY"]
        for t in largebinary_types:
            type_imports[t] = ("sqlalchemy", "LargeBinary")

        integer_types = ["SMALLINT", "INT", "INTEGER"]
        for t in integer_types:
            type_imports[t] = ("sqlalchemy", "Integer")

        numeric_types = ["DECIMAL", "NUMERIC"]
        for t in numeric_types:
            type_imports[t] = ("sqlalchemy", "Numeric")

        float_types = ["DOUBLE", "DOUBLE_PRECISION", "REAL"]
        for t in float_types:
            type_imports[t] = ("sqlalchemy", "Float")

        datetime_types = ["TIMESTAMP"]
        for t in datetime_types:
            type_imports[t] = ("sqlalchemy", "DateTime")

        self.render_module_variables(models)

        for model in models:
            if model.table.name in EXCLUDED_TABLES:
                continue
            table = model.table
            # schema = table.schema

            if table.schema and table.schema.startswith("pg_"):
                continue  # Skip Postgres System-Views
            if table.name.startswith("pg_"):
                continue  # Skip system views

            for col in table.columns:
                sa_type = sa_type_from_column(col)
                base_type = sa_type.split("(", 1)[0].strip()
                used_types.add(base_type)

            if table is not None and isinstance(model, ModelClass):
                self.base_class_name = "PortalObject"
                rendered.append(self.render_class(model))
            elif table is not None:
                rendered.append(f"{model.name} = {self.render_table(model.table)}")

        for typ in sorted(used_types):
            if typ in type_imports:
                module, name = type_imports[typ]
                self.add_literal_import(module, name)

        self.add_literal_import("sqlalchemy", "text")

        return "\n\n".join(rendered)
