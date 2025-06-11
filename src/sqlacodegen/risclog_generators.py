from typing import TYPE_CHECKING

from sqlalchemy import inspect

from sqlacodegen.generators import (
    Base,
    DeclarativeGenerator,
    LiteralImport,
    TablesGenerator,
)
from sqlacodegen.models import Model, ModelClass

if TYPE_CHECKING:
    from sqlacodegen.generators import TablesGenerator

VIEW_CLASS_TEMPLATE = '''\
class {classname}(PortalObject):
    __table__ = create_view(
        name='{table_name}',
        selectable=text("""{view_def}"""),
        metadata=PortalObject.metadata
    )
'''

BASE_META_DATA = Base(
    literal_imports=[
        LiteralImport(
            "risclog.sqlalchemy.model",
            "ObjectBase, declarative_base, class_registry",
        )
    ],
    declarations=[
        "PortalObject = declarative_base("
        "ObjectBase, class_registry=class_registry)"
    ],
    metadata_ref="PortalObject.metadata",
)


def clx_generate_base(self: "TablesGenerator") -> None:
    self.base = BASE_META_DATA


TablesGenerator.generate_base = clx_generate_base  # type: ignore[method-assign]


class DeclarativeGeneratorWithViews(DeclarativeGenerator):
    def generate_base(self) -> None:
        self.base = BASE_META_DATA

    def render_models(self, models: list[Model]) -> str:
        views_exist = False
        rendered: list[str] = []
        inspector = inspect(self.bind)
        schemas = set(table.schema for table in self.metadata.tables.values())
        views_by_schema = {
            schema: set(inspector.get_view_names(schema=schema)) for schema in schemas
        }

        self.add_literal_import("sqlalchemy", "text")
        self.render_module_variables(models)

        for model in models:
            table = model.table
            schema = table.schema
            schema_views = views_by_schema.get(schema, set())

            if table.name in schema_views:
                views_exist = True
                view_def = inspector.get_view_definition(table.name, schema)
                classname = (
                    "".join(x.capitalize() for x in table.name.split("_")) + "View"
                )
                code = VIEW_CLASS_TEMPLATE.format(
                    classname=classname,
                    table_name=table.name,
                    view_def=view_def.strip(),
                )
                rendered.append(code)
            elif isinstance(model, ModelClass):
                self.base_class_name = "PortalObject"
                rendered.append(self.render_class(model))
            else:
                rendered.append(f"{model.name} = {self.render_table(model.table)}")

        if views_exist:
            self.add_literal_import("sqlalchemy_utils.view", "create_view")

        return "\n\n".join(rendered)
