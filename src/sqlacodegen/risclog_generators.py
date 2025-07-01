import re
from pprint import pformat
from typing import TYPE_CHECKING, Any, Callable

from sqlalchemy import (
    Column,
    ForeignKeyConstraint,
    Index,
    PrimaryKeyConstraint,
    Table,
    UniqueConstraint,
    inspect,
    text,
    types,
)
from sqlalchemy import types as satypes
from sqlalchemy.engine import Connection, Engine

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

EXCLUDED_TABLES = {"tmp_functest", "accesslogfailed"}
INCLUDED_POLICY_ROLES = {"brokeruser"}
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
ALEMBIC_FUNCTION_TEMPLATE = """{varname} = PGFunction(
    schema={schema!r},
    signature={signature!r},
    definition=\"\"\"{definition}\"\"\",
)
"""
ALEMBIC_POLICIES_TEMPLATE = """{varname} = PGPolicy(
    schema={schema!r},
    signature={signature!r},
    definition=\"\"\"{definition}\"\"\",
    on_entity={on_entity!r},
)
"""
ALEMBIC_TRIGGER_TEMPLATE = """{varname} = PGTrigger(
    schema={schema!r},
    signature={signature!r},
    definition=\"\"\"{definition}\"\"\",
)
"""
ALEMBIC_AGGREGATE_TEMPLATE = """{varname} = PGAggregate(
    schema={schema!r},
    signature={signature!r},
    definition={definition!r},
    _sfunc={sfunc!r},
    _stype={stype!r},
    _initcond={initcond!r},
    _finalfunc={finalfunc!r},
)
"""
ALEMBIC_EXTENSION_TEMPLATE = """{varname} = PGExtension(
    schema={schema!r},
    signature={signature!r},
    definition={definition!r},
)
"""
ALEMBIC_SEQUENCE_TEMPLATE = """{varname} = PGSequence(
    schema={schema!r},
    signature={signature!r},
    definition=\"\"\"{definition}\"\"\",
)
"""

ALEMBIC_FUNCTION_STATEMENT = """SELECT
    pg_get_functiondef(p.oid) AS func
FROM
    pg_proc p
    JOIN pg_namespace n ON p.pronamespace = n.oid
WHERE
    p.prokind = 'f'
    AND n.nspname = 'public'
    AND n.nspname NOT IN ('pg_catalog', 'information_schema')
    -- Exclude functions that belong to extensions
    AND NOT EXISTS (
        SELECT 1
        FROM pg_depend d
        JOIN pg_extension e ON d.refobjid = e.oid
        WHERE d.objid = p.oid AND d.deptype = 'e'
    )
ORDER BY
    n.nspname,
    p.proname;
"""


ALEMBIC_POLICIES_STATEMENT = """SELECT
    pol.polname AS policy_name,
    ns.nspname AS schema_name,
    c.relname AS table_name,
    'FOR ' ||
      CASE pol.polcmd
        WHEN 'r' THEN 'SELECT'
        WHEN 'a' THEN 'ALL'
        WHEN 'w' THEN 'UPDATE'
        WHEN 'u' THEN 'UPDATE'
        WHEN 'd' THEN 'DELETE'
        WHEN 'i' THEN 'INSERT'
        ELSE '[' || pol.polcmd || ']'
      END
    || ' TO ' ||
      COALESCE(
        string_agg(r.rolname, ', ' ORDER BY r.rolname),
        'PUBLIC'
      ) AS for_to_clause,
    pol.polcmd,
    pg_get_expr(pol.polqual, pol.polrelid) AS using_clause,
    pg_get_expr(pol.polwithcheck, pol.polrelid) AS with_check
FROM
    pg_policy pol
    JOIN pg_class c ON pol.polrelid = c.oid
    JOIN pg_namespace ns ON ns.oid = c.relnamespace
    LEFT JOIN unnest(pol.polroles) AS r_oid ON TRUE
    LEFT JOIN pg_roles r ON r.oid = r_oid
WHERE
    ns.nspname = 'public'
GROUP BY pol.polname, ns.nspname, c.relname, pol.polcmd, pol.polrelid, pol.polqual, pol.polwithcheck
ORDER BY policy_name;  """

ALEMBIC_TRIGGER_STATEMENT = """SELECT
    trg.tgname AS trigger_name,
    tbl.relname AS table_name,
    pg_get_triggerdef(trg.oid, true) AS trigger_def
FROM
    pg_trigger trg
    JOIN pg_class tbl ON tbl.oid = trg.tgrelid
    JOIN pg_namespace ns ON ns.oid = tbl.relnamespace
WHERE
    NOT trg.tgisinternal
    AND ns.nspname = 'public';
"""
ALEMBIC_AGGREGATE_STATEMENT = """SELECT
    n.nspname AS schema,
    p.proname AS aggregate_name,
    pg_get_function_identity_arguments(p.oid) AS args,
    sf.sfunc_name AS sfunc,
    st.typname AS stype,
    ff.finalfunc_name AS finalfunc,
    a.agginitval AS initcond
FROM
    pg_proc p
    JOIN pg_namespace n ON p.pronamespace = n.oid
    JOIN pg_aggregate a ON a.aggfnoid = p.oid
    LEFT JOIN pg_type st ON a.aggtranstype = st.oid
    LEFT JOIN LATERAL (
        SELECT p1.proname AS sfunc_name
        FROM pg_proc p1
        WHERE p1.oid = a.aggtransfn
    ) sf ON TRUE
    LEFT JOIN LATERAL (
        SELECT p2.proname AS finalfunc_name
        FROM pg_proc p2
        WHERE p2.oid = a.aggfinalfn
    ) ff ON TRUE
WHERE
    n.nspname = :schema
    AND n.nspname NOT IN ('pg_catalog', 'information_schema')
    AND NOT EXISTS (
        SELECT 1
        FROM pg_depend d
        JOIN pg_extension e ON d.refobjid = e.oid
        WHERE d.objid = p.oid AND d.deptype = 'e'
    )
ORDER BY n.nspname, p.proname;
"""
ALEMBIC_EXTENSION_STATEMENT = """SELECT
    'CREATE EXTENSION IF NOT EXISTS ' || quote_ident(extname) ||
    ' WITH SCHEMA ' || quote_ident(nspname) || ';'
    AS create_extension_stmt,
    extname,
    nspname AS schema
FROM pg_extension
JOIN pg_namespace ON pg_extension.extnamespace = pg_namespace.oid
ORDER BY extname;
"""

ALEMBIC_SEQUENCE_STATEMENT = """
SELECT
    s.sequence_schema AS schema,
    s.sequence_name,
    s.data_type,
    s.start_value,
    s.minimum_value,
    s.maximum_value,
    s.increment,
    s.cycle_option AS cycle,
    ps.cache_size
FROM information_schema.sequences s
JOIN pg_catalog.pg_sequences ps
      ON s.sequence_schema = ps.schemaname
     AND s.sequence_name = ps.sequencename
WHERE s.sequence_schema NOT IN ('pg_catalog', 'information_schema')
ORDER BY s.sequence_schema, s.sequence_name;
"""


def finalize_alembic_utils(
    pg_alembic_definition: list[str],
    entities: list[str],
    entities_name: str | None,
) -> list[str]:
    imports = {
        "all_views": "from alembic_utils.pg_view import PGView",
        "all_functions": "from alembic_utils.pg_function import PGFunction",
        "all_policies": "from alembic_utils.pg_policy import PGPolicy",
        "all_triggers": "from alembic_utils.pg_trigger import PGTrigger",
        "all_sequences": "from alembic_utils.pg_sequence import PGSequence",
        "all_extensions": "from alembic_utils.pg_extension import PGExtension",
        "all_aggregates": "from alembic_utils.pg_aggregate import PGAggregate",
    }
    import_stmt = imports.get(
        entities_name or "all_views",
        "from alembic_utils.pg_view import PGView",
    )
    formatted = f"{entities_name} = [{', '.join(entities)}]\n"
    pg_alembic_definition.append(formatted)
    pg_alembic_definition.insert(0, f"{import_stmt}  # noqa: I001\n\n")

    return pg_alembic_definition


def parse_function_row(
    row: dict[str, Any], template_def: str, schema: str | None
) -> tuple[str, str | Any]:
    func_sql = row["func"]
    m = re.search(
        r"CREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION\s+([^(]+)\((.*?)\)",
        func_sql,
        re.IGNORECASE | re.DOTALL,
    )
    if not m:
        raise ValueError(f"Cannot extract function signature from: {func_sql[:100]}")
    name = m.group(1).strip().split(".")[-1]
    args = m.group(2).strip()
    signature = f"{name}({args})"

    m2 = re.search(
        r"CREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION\s+[^(]+\([^)]*\)\s*",
        func_sql,
        re.IGNORECASE | re.DOTALL,
    )
    if not m2:
        raise ValueError(f"Cannot extract function body from: {func_sql[:100]}")
    definition = func_sql[m2.end() :].strip()
    schema = schema or "public"

    name = name.lower()
    return template_def.format(
        varname=name,
        schema=schema,
        signature=signature,
        definition=unescape_sql_string(squash_whitespace(definition)),
    ), name


def parse_policy_row(
    policy: dict[str, Any], template_def: str, schema: str | None
) -> tuple[str, str] | None:
    policy_name = policy["policy_name"]
    schema_name = policy.get("schema_name", schema) or "public"
    table_name = policy["table_name"]

    for_to_clause = policy.get("for_to_clause", "")

    match = re.search(r"TO\s+([a-zA-Z0-9_,\s]+)", for_to_clause)
    if match:
        roles_raw = match.group(1)
        roles = [r.strip().lower() for r in roles_raw.split(",")]
        if any(role not in INCLUDED_POLICY_ROLES for role in roles):
            return None

    using = f" USING ({policy['using_clause']})" if policy.get("using_clause") else ""
    check = f" WITH CHECK ({policy['with_check']})" if policy.get("with_check") else ""

    definition = f"{for_to_clause}{using}{check}".strip()

    signature = f"{policy_name}.{table_name}"
    on_entity = f"{schema_name}.{table_name}"
    varname = f"{policy_name}_{table_name}".lower()

    code = template_def.format(
        varname=varname,
        schema=schema_name,
        signature=signature,
        definition=definition,
        on_entity=on_entity,
    )
    return code, varname


def parse_trigger_row(
    trigger: dict[str, str], template_def: str, schema: str | None
) -> tuple[str, str]:
    trigger_name = trigger["trigger_name"]
    table_name = trigger["table_name"]
    schema = schema or "public"

    varname = f"{trigger_name}_{table_name}".lower()
    signature = f"{trigger_name}.{table_name}"
    definition = trigger["trigger_def"].strip()
    definition = re.sub(r"\s+", " ", trigger["trigger_def"]).strip()

    code = template_def.format(
        varname=varname,
        schema=schema,
        signature=signature,
        definition=definition,
    )
    return code, varname


def parse_aggregate_row(
    row: dict[str, str], template_def: str, schema: str | None
) -> tuple[str, str]:
    aggregate_name = row["aggregate_name"]
    args = row.get("args", "")
    sfunc = row.get("sfunc")
    stype = row.get("stype")
    finalfunc = row.get("finalfunc")
    initcond = row.get("initcond")
    schema_val = schema or row.get("schema") or "public"

    # Baue die Definition als lesbare String-Config:
    definition_parts = []
    if sfunc:
        definition_parts.append(f"SFUNC = {sfunc}")
    if stype:
        definition_parts.append(f"STYPE = {stype}")
    if finalfunc:
        definition_parts.append(f"FINALFUNC = {finalfunc}")
    if initcond:
        definition_parts.append(f"INITCOND = {initcond}")
    definition = ", ".join(definition_parts)

    signature = f"{aggregate_name}({args})"
    varname = f"{aggregate_name}".lower()

    code = template_def.format(
        varname=varname,
        schema=schema_val,
        signature=signature,
        definition=definition,
        sfunc=sfunc,
        stype=stype,
        finalfunc=finalfunc,
        initcond=initcond,
    )
    return code, varname


def parse_extension_row(
    row: dict[str, Any], template_def: str, schema: str | None
) -> tuple[str, str]:
    definition = row["create_extension_stmt"].strip()
    signature = row["extname"]
    schema_val = row["schema"] or schema or "public"
    varname = f"{signature}_extension".lower()
    code = template_def.format(
        varname=varname, schema=schema_val, signature=signature, definition=definition
    )
    return code, varname


def parse_sequence_row(
    row: dict[str, Any], template_def: str, schema: str | None
) -> tuple[str, str]:
    schema_val = row["schema"] or schema or "public"
    signature = row["sequence_name"]
    varname = f"{signature}_sequence".lower()

    parts = [
        f"AS {row['data_type']}",
        f"START WITH {row['start_value']}",
        f"INCREMENT BY {row['increment']}",
        f"MINVALUE {row['minimum_value']}",
        f"MAXVALUE {row['maximum_value']}",
        f"CACHE {row['cache_size']}",
        "CYCLE"
        if str(row.get("cycle", "")).lower() in ("yes", "true", "on", "1")
        else "NO CYCLE",
    ]
    definition = "\n    ".join(parts)

    code = template_def.format(
        varname=varname,
        schema=schema_val,
        signature=signature,
        definition=definition,
    )
    return code, varname


def fetch_all_mappings(
    conn: Connection, sql: str, params: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    if params is None:
        params = {}
    close_after = False

    try:
        result = conn.execute(text(sql), params)
        return [dict(row) for row in result.mappings()]
    finally:
        if close_after:
            conn.close()


def squash_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def unescape_sql_string(s: str) -> str:
    return re.sub(r"\\(['\"])", r"\1", s)


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
    def generate_alembic_utils_entities(
        self,
        template: str,
        statement: str,
        parse_row_func: Callable[[dict[str, Any], str, str], str],
        entities_varname: str,
        schema: str = "public",
    ) -> list[str]:
        if isinstance(self.bind, Engine):
            conn = self.bind.connect()
        else:
            conn = self.bind

        if statement in globals():
            sql = globals()[statement]
        else:
            raise ValueError(f"Unknown statement: {statement}")

        if template in globals():
            template_def = globals()[template]
        else:
            raise ValueError(f"Unknown template: {template}")

        result: list[dict[str, Any]] = fetch_all_mappings(conn, sql, {"schema": schema})
        entities: list[str] = [
            parsed
            for row in result
            if (parsed := parse_row_func(row, template_def, schema)) is not None
        ]

        code = [code for code, _ in entities]  # type: ignore  # noqa: PGH003
        varnames = [varnames for _, varnames in entities]  # type: ignore  # noqa: PGH003
        return finalize_alembic_utils(
            code,
            varnames,
            entities_varname,
        )

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

        return "\n\n".join(rendered), finalize_alembic_utils(
            pg_alembic_definition, entities, entities_name
        ) if pg_alembic_definition else None
