from typing import TYPE_CHECKING, Any

from sqlalchemy import Column, Index, inspect, types as satypes

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
            enum_name = enum_class.__name__ if hasattr(enum_class, "__name__") else str(enum_class)
            return f"Enum({enum_name})"
        if hasattr(t, "enums"):
            return f"Enum({', '.join(repr(e) for e in t.enums)})"
        return "Enum"

    if isinstance(t, (satypes.FLOAT, satypes.Float, satypes.REAL)):
        return "Float"
    if isinstance(t, (satypes.INT, satypes.INTEGER, satypes.Integer, satypes.SMALLINT, satypes.SmallInteger)):
        return "Integer"
    if isinstance(t, (satypes.Interval,)):
        return "Interval"
    if isinstance(t, (satypes.JSON,)):
        return "JSON"
    if isinstance(t, (satypes.LargeBinary,)):
        return "LargeBinary"
    if isinstance(t, (satypes.NVARCHAR, satypes.VARCHAR, satypes.String, satypes.Text, satypes.TEXT, satypes.Unicode, satypes.UnicodeText)):
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
            if isinstance(item_type, (satypes.Integer, satypes.INT, satypes.SMALLINT, satypes.BigInteger)):
                return "ARRAY(Integer)"
            if isinstance(item_type, (satypes.String, satypes.Text, satypes.VARCHAR)):
                return "ARRAY(String)"
            if isinstance(item_type, (satypes.Numeric, satypes.DECIMAL, satypes.Float, satypes.DOUBLE, satypes.REAL)):
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



VIEW_CLASS_TEMPLATE = """\
class {classname}(PortalObject):
    __table__ = Table(
        "{table_name}", PortalObject.metadata,
{columns}
    )
    _sql_view_definition = text(\"\"\"{view_def}\"\"\")
"""

BASE_META_DATA = Base(
    literal_imports=[
        LiteralImport(
            "risclog.sqlalchemy.model",
            "ObjectBase, declarative_base, class_registry",
        )
    ],
    declarations=[
        "PortalObject = declarative_base(ObjectBase, class_registry=class_registry)"
    ],
    metadata_ref="PortalObject.metadata",
)


def clx_generate_base(self: "TablesGenerator") -> None:
    self.base = BASE_META_DATA


TablesGenerator.generate_base = clx_generate_base  # type: ignore[method-assign]


def clx_render_index(self: "TablesGenerator", index: Index) -> str:
    extra_args = [repr(col.name) for col in index.columns]
    if not extra_args:
        name = index.name if index is not None else "<unnamed>"
        table_name = index.table.name if index.table is not None else "<no-table>"
        print(
            f"# WARNING: Skipped index '{name}' on table '{table_name}' because it has no columns."
        )
        return ""
    kwargs = {}
    if index.unique:
        kwargs["unique"] = True
    return render_callable("Index", repr(index.name), *extra_args, kwargs=kwargs)


TablesGenerator.render_index = clx_render_index  # type: ignore[method-assign]


class DeclarativeGeneratorWithViews(DeclarativeGenerator):
    def generate_base(self) -> None:
        self.base = BASE_META_DATA

    def render_view_class(self, model: Model) -> str:
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

        self.add_literal_import("sqlalchemy.dialects.postgresql", "UUID as SA_UUID")
        self.add_literal_import("uuid", "uuid4")

        result: str = VIEW_CLASS_TEMPLATE.format(
            classname=classname,
            table_name=table.name,
            columns=columns_code,
            view_def=view_def,
        )
        return result


    def render_models(self, models: list[Model]) -> str:
        rendered: list[str] = []
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
            # "Uuid": ("sqlalchemy.dialects.postgresql", "UUID as SA_UUID"),
        }

        string_types = ["CHAR", "NCHAR", "NVARCHAR", "UNICODE", "UNICODETEXT", "TEXT", "VARCHAR"]
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
                code = self.render_view_class(model)
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
        return "\n\n".join(rendered)
