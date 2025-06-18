import re
from pprint import pformat
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Column,
    ForeignKeyConstraint,
    Index,
    PrimaryKeyConstraint,
    Table,
    UniqueConstraint,
    inspect,
    types,
)
from sqlalchemy import types as satypes

from sqlacodegen.generators import (
    Base,
    DeclarativeGenerator,
    LiteralImport,
    TablesGenerator,
)
from sqlacodegen.models import Model, ModelClass
from sqlacodegen.utils import (
    get_constraint_sort_key,
    render_callable,
    uses_default_name,
)

if TYPE_CHECKING:
    from sqlacodegen.generators import TablesGenerator

EXCLUDED_TABLES = {"tmp_functest"}
BASE_META_DATA = Base(
    literal_imports=[
        LiteralImport(
            "risclog.claimxdb.database",
            "PortalObject",
        )
    ],
    declarations=[],
    metadata_ref="PortalObject.metadata",
)
ORM_VIEW_CLASS_TEMPLATE = """\
class {classname}(PortalObject):
    __table__ = Table(
        "{table_name}", PortalObject.metadata,
{columns}
    )
"""
ALEMBIC_VIEW_CLASS_TEMPLATE = """{varname} = PGView(
    schema={schema!r},
    signature={signature!r},
    definition=\"\"\"{definition}\"\"\",
)
"""


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


def clx_generate_base(self: "TablesGenerator") -> None:
    self.base = BASE_META_DATA


TablesGenerator.generate_base = clx_generate_base  # type: ignore[method-assign]


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


def clx_render_table(self: "TablesGenerator", table: Table) -> str:
    args: list[str] = [f"{table.name!r}, {self.base.metadata_ref}"]
    kwargs: dict[str, object] = {}
    for column in table.columns:
        args.append(self.render_column(column, True, is_table=True))

    for constraint in sorted(table.constraints, key=get_constraint_sort_key):
        if uses_default_name(constraint):
            if isinstance(constraint, PrimaryKeyConstraint):
                continue
            elif isinstance(constraint, (ForeignKeyConstraint, UniqueConstraint)):
                if len(constraint.columns) == 1:
                    continue
        args.append(self.render_constraint(constraint))

    for index in sorted(table.indexes, key=lambda i: str(i.name or "")):
        if len(index.columns) > 1 or not uses_default_name(index):
            idx_code = self.render_index(index)
            if idx_code.strip() and idx_code is not None:
                args.append(idx_code)

    if table.schema:
        kwargs["schema"] = repr(table.schema)

    table_comment = getattr(table, "comment", None)
    if table_comment:
        kwargs["comment"] = repr(table.comment)

    return render_callable("Table", *args, kwargs=kwargs, indentation="    ")


TablesGenerator.render_table = clx_render_table  # type: ignore[method-assign]


def clx_generate(self: "TablesGenerator") -> tuple[str, list[str] | None]:
    self.generate_base()

    sections: list[str] = []

    # Remove unwanted elements from the metadata
    for table in list(self.metadata.tables.values()):
        if self.should_ignore_table(table):
            self.metadata.remove(table)
            continue

        if "noindexes" in self.options:
            table.indexes.clear()

        if "noconstraints" in self.options:
            table.constraints.clear()

        if "nocomments" in self.options:
            table.comment = None

        for column in table.columns:
            if "nocomments" in self.options:
                column.comment = None

    # Use information from column constraints to figure out the intended column
    # types
    for table in self.metadata.tables.values():
        self.fix_column_types(table)

    # Generate the models
    models: list[Model] = self.generate_models()

    # Render module level variables
    variables = self.render_module_variables(models)
    if variables:
        sections.append(variables + "\n")

    # Render models
    rendered_models, pg_alembic_definition = self.render_models(models)  # type: ignore[misc]

    if rendered_models:  # type: ignore[has-type]
        sections.append(rendered_models)  # type: ignore[has-type]

    # Render collected imports
    groups = self.group_imports()
    imports = "\n\n".join("\n".join(line for line in group) for group in groups)
    if imports:
        sections.insert(0, imports)

    return "\n\n".join(sections) + "\n", pg_alembic_definition  # type: ignore[has-type]


TablesGenerator.generate = clx_generate  # type: ignore[assignment]


class DeclarativeGeneratorWithViews(DeclarativeGenerator):
    def render_view_classes(
        self, model: Model, signature: str, schema: str
    ) -> tuple[str, str]:
        table = model.table
        classname = "".join(x.capitalize() for x in table.name.split("_"))
        if not classname.lower().endswith("view"):
            classname += "View"

        columns = []
        has_id = False

        for col in table.columns:
            if col.name == "id":
                has_id = True

                columns.append(
                    " " * 8
                    + "Column('id', SA_UUID(as_uuid=True), primary_key=True, default=uuid4)"
                )
            else:
                sa_type = sa_type_from_column(col)
                columns.append(f"{' ' * 0}Column('{col.name}', {sa_type})")

        if not has_id:
            columns.insert(
                0,
                " " * 8
                + "Column('id', SA_UUID(as_uuid=True), primary_key=True, default=uuid4)",
            )

        columns_code = ",\n        ".join(columns)

        view_def = getattr(model, "view_definition", None)
        if not view_def:
            inspector = inspect(self.bind)
            view_def = inspector.get_view_definition(table.name, table.schema)
        view_def = (view_def or "").strip()
        view_def = re.sub(r"\s+", " ", view_def).strip()

        self.add_literal_import("sqlalchemy.dialects.postgresql", "UUID as SA_UUID")
        self.add_literal_import("uuid", "uuid4")

        orm_result: str = ORM_VIEW_CLASS_TEMPLATE.format(
            classname=classname,
            table_name=table.name,
            columns=columns_code,
        )
        alembic_result = ALEMBIC_VIEW_CLASS_TEMPLATE.format(
            varname=signature,
            schema=schema,
            signature=signature,
            definition=view_def,
        )
        return orm_result, alembic_result

    def render_table_args(self, table: Table) -> str:
        args: list[str] = []
        kwargs: dict[str, str] = {}

        for constraint in sorted(table.constraints, key=get_constraint_sort_key):
            if uses_default_name(constraint):
                if isinstance(constraint, PrimaryKeyConstraint):
                    continue
                if (
                    isinstance(constraint, (ForeignKeyConstraint, UniqueConstraint))
                    and len(constraint.columns) == 1
                ):
                    continue
            args.append(self.render_constraint(constraint))

        for index in sorted(table.indexes, key=lambda i: str(i.name or "")):
            if len(index.columns) > 1 or not uses_default_name(index):
                idx_code = self.render_index(index)
                if idx_code.strip() and idx_code is not None:
                    args.append(idx_code)

        if table.schema:
            kwargs["schema"] = table.schema

        if table.comment:
            kwargs["comment"] = table.comment

        if kwargs:
            formatted_kwargs = pformat(kwargs)
            if not args:
                return formatted_kwargs
            else:
                args.append(formatted_kwargs)

        if args:
            rendered_args = f",\n{self.indentation}".join(args)
            if len(args) == 1:
                rendered_args += ","
            return f"(\n{self.indentation}{rendered_args}\n)"
        else:
            return ""

    def generate_base(self) -> None:
        self.base = BASE_META_DATA

    def finalize_alembic_utils(
        self,
        pg_alembic_definition: list[str],
        entities: list[str],
        entities_name: str | None,
    ) -> list[str]:
        imports = {
            "all_views": "from alembic_utils.pg_view import PGView  # noqa: I001"
        }
        import_stmt = imports.get(
            entities_name or "all_views",
            "from alembic_utils.pg_view import PGView  # noqa: I001",
        )
        formatted = f"{entities_name} = [{', '.join(entities)}]\n"
        pg_alembic_definition.append(formatted)
        pg_alembic_definition.insert(0, f"{import_stmt}\n\n")

        return pg_alembic_definition

    def render_models(self, models: list[Model]) -> tuple[str, list[str] | None]:  # type: ignore[override]
        rendered: list[str] = []
        pg_alembic_definition = []
        entities = []
        entities_name = None
        inspector = inspect(self.bind)
        schemas = set(table.schema for table in self.metadata.tables.values())
        views_by_schema = {
            schema: set(inspector.get_view_names(schema=schema)) for schema in schemas
        }
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
            schema = table.schema
            schema_views = views_by_schema.get(schema, set())

            if table.schema and table.schema.startswith("pg_"):
                continue  # Skip Postgres System-Views
            if table.name.startswith("pg_"):
                continue  # Skip system views

            for col in table.columns:
                sa_type = sa_type_from_column(col)
                base_type = sa_type.split("(", 1)[0].strip()
                used_types.add(base_type)

            if table is not None and table.name in schema_views:
                code, pg_alembic = self.render_view_classes(
                    model, table.name, schema or "public"
                )
                pg_alembic_definition.append(pg_alembic)
                entities.append(table.name)
                entities_name = "all_views"
                rendered.append(code)
                self.add_literal_import("sqlalchemy", "Column")
            elif table is not None and isinstance(model, ModelClass):
                self.base_class_name = "PortalObject"
                rendered.append(self.render_class(model))
            elif table is not None:
                rendered.append(f"{model.name} = {self.render_table(model.table)}")

        for typ in sorted(used_types):
            if typ in type_imports:
                module, name = type_imports[typ]
                self.add_literal_import(module, name)

        self.add_literal_import("sqlalchemy", "text")

        return "\n\n".join(rendered), self.finalize_alembic_utils(
            pg_alembic_definition, entities, entities_name
        ) if pg_alembic_definition else None
