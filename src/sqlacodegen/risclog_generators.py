from typing import TYPE_CHECKING, Any

from sqlalchemy import Column, Index, inspect

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
    try:
        if hasattr(t, "python_type"):
            py = t.python_type
            if py is int:
                return "Integer"
            elif py is bool:
                return "Boolean"
            elif py is float:
                return "Numeric"
            elif py is str:
                return "String"
    except NotImplementedError:
        pass

    t_str = str(t).upper()
    if "NUMERIC" in t_str or "DECIMAL" in t_str:
        return "Numeric"
    if "FLOAT" in t_str or "DOUBLE" in t_str or "REAL" in t_str:
        return "Float"
    if "DATE" in t_str and "TIME" not in t_str:
        return "Date"
    if "DATETIME" in t_str or "TIMESTAMP" in t_str:
        return "DateTime"
    if "TIME" in t_str and "STAMP" not in t_str:
        return "Time"
    if "BOOLEAN" in t_str or "BOOL" in t_str:
        return "Boolean"
    if "SMALLINT" in t_str:
        return "SmallInteger"
    if "BIGINT" in t_str:
        return "BigInteger"
    if "INT" in t_str:
        return "Integer"
    if "UUID" in t_str:
        return "Uuid"
    if "ARRAY" in t_str:
        return "ARRAY"
    if "JSONB" in t_str or "JSON" in t_str:
        return "JSON"
    if "BYTEA" in t_str or "BLOB" in t_str:
        return "LargeBinary"
    if "CHAR" in t_str or "TEXT" in t_str or "STRING" in t_str or "CITEXT" in t_str:
        return "String"
    if "OID" in t_str:
        return "OID"

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

        self.add_literal_import("sqlalchemy", "text")
        self.add_literal_import("sqlalchemy", "Table")
        self.add_literal_import("sqlalchemy", "Column")
        self.add_literal_import("sqlalchemy", "Integer")
        self.add_literal_import("sqlalchemy", "String")
        self.add_literal_import("sqlalchemy", "Boolean")
        self.add_literal_import("sqlalchemy", "Date")
        self.add_literal_import("sqlalchemy", "Numeric")
        self.render_module_variables(models)

        for model in models:
            table = model.table
            schema = table.schema
            schema_views = views_by_schema.get(schema, set())

            if table is not None and table.name in schema_views:
                code = self.render_view_class(model)
                rendered.append(code)
            elif table is not None and isinstance(model, ModelClass):
                self.base_class_name = "PortalObject"
                rendered.append(self.render_class(model))
            elif table is not None:
                rendered.append(f"{model.name} = {self.render_table(model.table)}")

        return "\n\n".join(rendered)
