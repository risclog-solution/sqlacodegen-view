import re
from pprint import pformat
from typing import TYPE_CHECKING, Any, Callable, cast

from sqlalchemy import (
    Column,
    Computed,
    DefaultClause,
    ForeignKeyConstraint,
    Identity,
    Index,
    MetaData,
    PrimaryKeyConstraint,
    Table,
    UniqueConstraint,
    inspect,
    text,
)
from sqlalchemy import types as satypes
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.sql.elements import TextClause
from sqlalchemy.sql.functions import next_value

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
INCLUDED_POLICY_ROLES = {"brokeruser", " clx_readonly", "clx"}
BASE_META_DATA = Base(
    literal_imports=[
        LiteralImport(
            "risclog.claimxdb.database.base",
            "PortalObject",
        )
    ],
    declarations=[],
    metadata_ref="PortalObject.metadata",
)
ORM_VIEW_CLASS_TEMPLATE = """\
class {classname}(PortalObject):  # type: ignore[misc]
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
    enable_rls={enable_rls}
)
"""
ALEMBIC_TRIGGER_TEMPLATE = """{varname} = PGTrigger(
    schema={schema!r},
    signature={signature!r},
    definition=\"\"\"{definition}\"\"\",
    on_entity={on_entity!r},
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
)
"""
ALEMBIC_SEQUENCE_TEMPLATE = """{varname} = PGSequence(
    schema={schema!r},
    signature={signature!r},
    definition=\"\"\"{definition}\"\"\",
)
"""
ALEMBIC_PUBLICATION_TEMPLATE = """{varname} = PGPublication(
    name={name!r},
    tables={tables!r},
    publish={publish!r},
)
"""
ALEMBIC_PUBLICATION_STATEMENT = """
SELECT
    p.pubname,
    array_remove(array_agg(pt.relname), NULL) as tables,
    (
      CASE WHEN p.pubinsert THEN 'insert' ELSE '' END ||
      CASE WHEN p.pubupdate THEN ', update' ELSE '' END ||
      CASE WHEN p.pubdelete THEN ', delete' ELSE '' END ||
      CASE WHEN p.pubtruncate THEN ', truncate' ELSE ''
    END
    ) as publish
FROM
    pg_publication p
    LEFT JOIN pg_publication_rel pr ON pr.prpubid = p.oid
    LEFT JOIN pg_class pt ON pt.oid = pr.prrelid
GROUP BY p.pubname, p.pubinsert, p.pubupdate, p.pubdelete, p.pubtruncate
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
WHERE nspname != 'pg_catalog'
ORDER BY extname;
"""

ALEMBIC_STANDALONE_SEQUENCE_STATEMENT = """
WITH seqs AS (
  SELECT c.oid AS seq_oid,
         n.nspname AS schema,
         c.relname AS sequence_name
  FROM pg_class c
  JOIN pg_namespace n ON n.oid = c.relnamespace
  WHERE c.relkind = 'S'
    AND n.nspname NOT IN ('pg_catalog','information_schema')
    AND n.nspname NOT LIKE 'pg_toast%'
    AND n.nspname NOT LIKE 'pg_temp_%'
)
SELECT
  s.schema,
  s.sequence_name,
  ps.data_type,
  ps.start_value,
  ps.min_value     AS minimum_value,
  ps.max_value     AS maximum_value,
  ps.increment_by  AS increment,
  ps.cycle,
  ps.cache_size
FROM seqs s
JOIN pg_catalog.pg_sequences ps
  ON ps.schemaname   = s.schema
 AND ps.sequencename = s.sequence_name
LEFT JOIN pg_depend owned
       ON owned.classid = 'pg_class'::regclass
      AND owned.objid   = s.seq_oid
      AND owned.deptype IN ('a','i')   -- serial/identity ownership
LEFT JOIN pg_depend used
       ON used.refclassid = 'pg_class'::regclass
      AND used.refobjid   = s.seq_oid
      AND used.classid    = 'pg_attrdef'::regclass  -- used in a column DEFAULT
WHERE owned.objid IS NULL
  AND used.objid  IS NULL
ORDER BY s.schema, s.sequence_name;
"""



def parse_publication_row(
    row: dict[str, Any],
    template_def: str,
    schema: str | None,
) -> tuple[str, str] | None:
    name = row.get("pubname")
    tables = row.get("tables") or []
    if isinstance(tables, str):
        tables = [t.strip() for t in tables.split(",") if t.strip()]
    publish = row.get("publish") or ""
    owner = row.get("owner") or None

    varname = f"{name}".lower()
    code = template_def.format(
        varname=varname,
        name=name,
        tables=tables,
        publish=publish,
        owner=owner,
    )
    return code, varname


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
        "all_publications": "from risclog.claimxdb.alembic.object_ops import PGPublication",
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
    return (
        template_def.format(
            varname=name,
            schema=schema,
            signature=signature,
            definition=unescape_sql_string(squash_whitespace(definition)),
        ),
        name,
    )


def parse_policy_row(
    policy: dict[str, Any],
    template_def: str,
    schema: str | None,
    rls_enabled_tables: set[tuple[str, str]],
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

    signature = f"{policy_name}_{table_name}"
    on_entity = f"{schema_name}.{table_name}"
    varname = f"{policy_name}_{table_name}".lower()

    table_key = (schema_name, table_name)
    enable_rls = False
    if table_key not in rls_enabled_tables:
        enable_rls = True
        rls_enabled_tables.add(table_key)

    code = template_def.format(
        varname=varname,
        schema=schema_name,
        signature=signature,
        definition=definition,
        on_entity=on_entity,
        enable_rls=enable_rls,
    )
    return code, varname


def parse_trigger_row(
    trigger: dict[str, str], template_def: str, schema: str | None
) -> tuple[str, str]:
    trigger_name = trigger["trigger_name"]
    table_name = trigger["table_name"]
    schema_val = schema or "public"

    varname = f"{trigger_name}_{table_name}".lower()
    signature = trigger_name

    def extract_pgtrigger_definition(trigger_def: str, trigger_name: str) -> str:
        m = re.search(
            rf"CREATE\s+TRIGGER\s+{re.escape(trigger_name)}\s+(.*)",
            trigger_def,
            re.IGNORECASE | re.DOTALL,
        )
        if m:
            return m.group(1).strip()
        return trigger_def.strip()

    definition = extract_pgtrigger_definition(trigger["trigger_def"], trigger_name)
    on_entity = f"{schema_val}.{table_name}"

    code = template_def.format(
        varname=varname,
        schema=schema_val,
        signature=signature,
        definition=definition,
        on_entity=on_entity,
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
    varname = signature.lower()

    parts = [
        f"AS {row['data_type']}",
        f"START WITH {row['start_value']}",
        f"INCREMENT BY {row['increment']}",
        f"MINVALUE {row['minimum_value']}",
        f"MAXVALUE {row['maximum_value']}",
        f"CACHE {row['cache_size']}",
        (
            "CYCLE"
            if str(row.get("cycle", "")).lower() in ("yes", "true", "on", "1")
            else "NO CYCLE"
        ),
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


def unqualify(colname: str) -> str:
    if isinstance(colname, str):
        return colname.split(".")[-1]
    return str(colname)


def clx_render_index(self: "TablesGenerator", index: Index) -> str:
    from sqlalchemy.sql.elements import TextClause

    args = [repr(index.name)]
    kwargs: dict[str, Any] = {}
    opclass_map = {}

    # --- Columns ---
    if getattr(index, "columns", None) and len(index.columns) > 0:
        for col in index.columns:
            args.append(repr(unqualify(col.name)))
            # Operator-Class GIN/TRGM
            if (
                "postgresql" in index.dialect_options
                and index.dialect_options["postgresql"].get("using") == "gin"
            ):
                coltype = getattr(col.type, "python_type", None)
                if isinstance(
                    col.type, (satypes.String, satypes.Text, satypes.Unicode)
                ) or (coltype and coltype is str):
                    opclass_map[unqualify(col.name)] = "gin_trgm_ops"
    # --- Expressions/TextClause ---
    elif getattr(index, "expressions", None) and len(index.expressions) > 0:
        for expr in index.expressions:
            if isinstance(expr, TextClause):
                expr_str = str(expr)
                # GIN/TRGM als Suffix
                if (
                    "postgresql" in index.dialect_options
                    and index.dialect_options["postgresql"].get("using") == "gin"
                    and not expr_str.rstrip().endswith("gin_trgm_ops")
                ):
                    expr_str = f"{expr_str} gin_trgm_ops"
                args.append(f"text({expr_str!r})")
            else:
                expr_str = str(expr)
                m = re.match(r"^upper\(\((\w+)\)::text\)$", expr_str)
                if (
                    m
                    and "postgresql" in index.dialect_options
                    and index.dialect_options["postgresql"].get("using") == "gin"
                ):
                    args.append(f"text('upper(({m.group(1)})::text) gin_trgm_ops')")
                else:
                    args.append(f"text({expr_str!r})")
    else:
        # Fallback
        pass

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

    return render_callable("Index", *args, kwargs=kwargs)


TablesGenerator.render_index = clx_render_index  # type: ignore[method-assign]


def clx_render_table(self: "TablesGenerator", table: Table) -> str:
    args: list[str] = [f"{table.name!r}, {self.base.metadata_ref}"]
    kwargs: dict[str, object] = {}

    # Columns
    for column in table.columns:
        args.append(self.render_column(column, True, is_table=True))

    # Constraints
    for constraint in sorted(table.constraints, key=get_constraint_sort_key):
        if uses_default_name(constraint):
            if isinstance(constraint, PrimaryKeyConstraint):
                continue
            elif isinstance(constraint, (ForeignKeyConstraint, UniqueConstraint)):
                if len(constraint.columns) == 1:
                    continue
        args.append(self.render_constraint(constraint))

    # Indices
    for index in sorted(table.indexes, key=lambda i: str(i.name or "")):
        orig_columns = getattr(index, "columns", [])
        if orig_columns:
            table.indexes.remove(index)
            columns = [table.c[unqualify(col.name)] for col in orig_columns]
            new_index = Index(index.name, *columns, **index.kwargs)
            table.append_constraint(new_index)
        idx_code = self.render_index(index)
        if idx_code.strip() and idx_code is not None:
            args.append(idx_code)

    if table.schema:
        kwargs["schema"] = table.schema

    # Table comment
    table_comment = getattr(table, "comment", None)
    if table_comment:
        kwargs["comment"] = repr(table_comment)

    return render_callable("Table", *args, kwargs=kwargs, indentation="    ")


TablesGenerator.render_table = clx_render_table  # type: ignore[method-assign]


def clx_render_column(
    self: "TablesGenerator",
    column: Column[Any],
    show_name: bool,
    is_table: bool = False,
) -> str:
    args = []
    kwargs: dict[str, Any] = {}
    kwarg = []
    is_sole_pk = column.primary_key and len(column.table.primary_key) == 1
    dedicated_fks = [
        c
        for c in column.foreign_keys
        if c.constraint
        and len(c.constraint.columns) == 1
        and uses_default_name(c.constraint)
    ]
    is_unique = any(
        isinstance(c, UniqueConstraint)
        and set(c.columns) == {column}
        and uses_default_name(c)
        for c in column.table.constraints
    )
    is_unique = is_unique or any(
        i.unique and set(i.columns) == {column} and uses_default_name(i)
        for i in column.table.indexes
    )
    is_primary = (
        any(
            isinstance(c, PrimaryKeyConstraint)
            and column.name in c.columns
            and uses_default_name(c)
            for c in column.table.constraints
        )
        or column.primary_key
    )
    has_index = any(
        set(i.columns) == {column} and uses_default_name(i)
        for i in column.table.indexes
    )

    if show_name:
        args.append(repr(column.name))

    # Render the column type if there are no foreign keys on it or any of them
    # points back to itself
    if not dedicated_fks or any(fk.column is column for fk in dedicated_fks):
        args.append(self.render_column_type(column.type))

    for fk in dedicated_fks:
        args.append(self.render_constraint(fk))

    if column.default is not None:
        args.append(repr(column.default))

    if column.key != column.name:
        kwargs["key"] = column.key
    if is_primary:
        kwargs["primary_key"] = True
    if not column.nullable and not is_sole_pk and is_table:
        kwargs["nullable"] = False

    if is_unique:
        column.unique = True
        kwargs["unique"] = True
    if has_index:
        column.index = True
        kwarg.append("index")
        kwargs["index"] = True

    # --- SERVER DEFAULT HANDLING ---
    if isinstance(column.server_default, DefaultClause):
        kwargs["server_default"] = render_callable(
            "text", repr(cast(TextClause, column.server_default.arg).text)
        )
    elif isinstance(column.server_default, Computed):
        expression = str(column.server_default.sqltext)

        computed_kwargs = {}
        if column.server_default.persisted is not None:
            computed_kwargs["persisted"] = column.server_default.persisted

        args.append(
            render_callable("Computed", repr(expression), kwargs=computed_kwargs)
        )
    elif isinstance(column.server_default, Identity):
        args.append(repr(column.server_default))
    elif isinstance(column.server_default, next_value):
        # --------- NEU: Sequence/next_value ----------
        seq = column.server_default.sequence
        if seq is not None:
            default_schema = "public"
            kwargs["server_default"] = (
                f"Sequence({seq.name!r}{f', schema={seq.schema!r}' if seq.schema else f', schema={default_schema!r}'}).next_value()"
            )
        else:
            kwargs["server_default"] = "None  # Sequence not detected"
    elif column.server_default is not None:
        kwargs["server_default"] = repr(column.server_default)
    # -----------------------------------------------

    comment = getattr(column, "comment", None)
    if comment:
        kwargs["comment"] = repr(comment)

    return self.render_column_callable(is_table, *args, **kwargs)


TablesGenerator.render_column = clx_render_column  # type: ignore[method-assign]


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


def get_table_managed_sequences(metadata: MetaData) -> set[str]:
    seq_names = set()
    for table in metadata.tables.values():
        for column in table.columns:
            default = getattr(column, "default", None)
            if default is not None:
                if hasattr(default, "name"):
                    seq_names.add(default.name)
            if hasattr(column, "sequence") and column.sequence is not None:
                seq_names.add(column.sequence.name)
    return seq_names


class DeclarativeGeneratorWithViews(DeclarativeGenerator):
    def generate_alembic_utils_sequences(
        self,
        template: str,
        statement: str,
        parse_row_func: Callable[..., tuple[str, str] | None],
        schema: str = "public",
        entities_varname: str = "all_sequences",
    ) -> list[str]:
        if isinstance(self.bind, Engine):
            conn = self.bind.connect()
        else:
            conn = self.bind

        sql = globals()[statement]
        template_def = globals()[template]
        result: list[dict[str, Any]] = fetch_all_mappings(conn, sql, {"schema": schema})

        entities = [
            parsed
            for row in result
            if (parsed := parse_row_func(row, template_def, schema)) is not None
        ]

        code = [code for code, _ in entities]
        varnames = [varname for _, varname in entities]
        return finalize_alembic_utils(code, varnames, entities_varname)

    def generate_alembic_utils_entities(
        self,
        template: str,
        statement: str,
        parse_row_func: Callable[..., tuple[str, str] | None],
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
        entities: list[tuple[str, str]]

        if parse_row_func.__name__ == "parse_policy_row":
            rls_enabled_tables: set[tuple[str, str]] = set()
            entities = [
                parsed
                for row in result
                if (
                    parsed := parse_row_func(
                        row, template_def, schema, rls_enabled_tables
                    )
                )
                is not None
            ]
        else:
            entities = [
                parsed
                for i, row in enumerate(result)
                if (parsed := parse_row_func(row, template_def, schema)) is not None
            ]

        code = [code for code, _ in entities]
        varnames = [varname for _, varname in entities]
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
        if not classname.endswith("View"):
            classname += "View"

        columns = []
        has_id = False

        for col in table.columns:
            sa_type = sa_type_from_column(col)
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

        # NEU: ALLE Indexe (egal ob "special" oder nicht)
        for index in sorted(table.indexes, key=lambda i: str(i.name or "")):
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

        def get_extension_object_names(
            conn: Connection, schemas: set[str | None]
        ) -> Any:
            extension_objs = set()
            for schema in schemas:
                result = conn.execute(
                    text(
                        """
                    SELECT c.relname
                    FROM pg_class c
                    JOIN pg_namespace n ON c.relnamespace = n.oid
                    WHERE n.nspname = :schema
                    AND c.relkind IN ('r','v')
                    AND EXISTS (
                        SELECT 1 FROM pg_depend d
                        JOIN pg_extension e ON d.refobjid = e.oid
                        WHERE d.objid = c.oid AND d.deptype = 'e'
                    )
                """
                    ),
                    {"schema": schema},
                )
                extension_objs |= {row[0] for row in result}
            return extension_objs

        conn = self.bind.connect() if hasattr(self.bind, "connect") else self.bind
        EXTENSION_OBJECTS = get_extension_object_names(conn, schemas)

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
        all = []
        for model in models:
            table = model.table
            if table.name in EXCLUDED_TABLES:
                continue
            schema = table.schema
            schema_views = views_by_schema.get(schema, set())

            if table.schema and table.schema.startswith("pg_"):
                continue
            if table.name.startswith("pg_"):
                continue
            if table.name in EXTENSION_OBJECTS:
                continue

            for col in table.columns:
                sa_type = sa_type_from_column(col)
                base_type = sa_type.split("(", 1)[0].strip()
                used_types.add(base_type)

            if table is not None and table.name in schema_views:
                code, pg_alembic = self.render_view_classes(
                    model, table.name, schema or "public"
                )
                classname = "".join(x.capitalize() for x in table.name.split("_"))
                if not classname.endswith("View"):
                    classname += "View"
                all.append(classname)

                pg_alembic_definition.append(pg_alembic)
                entities.append(table.name)
                entities_name = "all_views"
                rendered.append(code)
                self.add_literal_import("sqlalchemy", "Column")
            elif table is not None and isinstance(model, ModelClass):
                self.base_class_name = "PortalObject"
                all.append(model.name)

                rendered.append(self.render_class(model))
            elif table is not None:
                rendered.append(f"{model.name} = {self.render_table(model.table)}")

        rendered.append(f"__all__ = {all}")
        for typ in sorted(used_types):
            if typ in type_imports:
                module, name = type_imports[typ]
                self.add_literal_import(module, name)

        self.add_literal_import("sqlalchemy", "text")
        self.add_literal_import("sqlalchemy", "FetchedValue")

        return "\n\n".join(rendered), (
            finalize_alembic_utils(pg_alembic_definition, entities, entities_name)
            if pg_alembic_definition
            else None
        )
