from fastapi import FastAPI, HTTPException, Query, Request, Path, Body
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, MetaData, Table, text
from sqlalchemy.engine.reflection import Inspector
import sqlalchemy
import os
import re

app = FastAPI(docs_url="/swagger")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://usuario:clave@localhost:3306/tu_base"
)

engine = create_engine(DATABASE_URL)
metadata = MetaData()
metadata.reflect(bind=engine)
inspector = Inspector.from_engine(engine)

# ----------------------------------------
# ODATA $filter advanced parser (safe)
# ----------------------------------------

def tokenize_filter(s):
    # Simple lexer for OData expressions (only the supported subset)
    token_specification = [
        ('LPAREN', r'\('),
        ('RPAREN', r'\)'),
        ('AND', r'\band\b'),
        ('OR', r'\bor\b'),
        ('OP', r'\b(eq|ne|gt|ge|lt|le)\b'),
        ('CONTAINS', r'contains'),
        ('COMMA', r','),
        ('STRING', r"'([^']*)'"),
        ('NUMBER', r'-?\d+(\.\d+)?'),
        ('ID', r'[A-Za-z_][A-Za-z0-9_]*'),
        ('SKIP', r'[ \t]+'),
        ('MISMATCH', r'.'),
    ]
    tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
    for mo in re.finditer(tok_regex, s):
        kind = mo.lastgroup
        value = mo.group()
        if kind == 'SKIP':
            continue
        elif kind == 'STRING':
            value = mo.group(1)
        elif kind == 'NUMBER':
            value = float(value) if '.' in value else int(value)
        elif kind == 'MISMATCH':
            raise RuntimeError(f'Unexpected character: {value}')
        yield (kind, value)

def parse_filter_expr(tokens, columns, param_idx=0):
    def peek():
        return tokens[0] if tokens else None
    def pop():
        return tokens.pop(0) if tokens else None

    def parse_atom():
        nonlocal param_idx
        tok = peek()
        if not tok:
            raise RuntimeError("Unexpected end of filter expression")
        kind, value = tok
        if kind == 'LPAREN':
            pop()
            expr, params = parse_expr()
            if not tokens or tokens[0][0] != 'RPAREN':
                raise RuntimeError("Expected ')'")
            pop()
            return f"({expr})", params
        elif kind == 'CONTAINS':
            pop()
            if not tokens or tokens[0][0] != 'LPAREN':
                raise RuntimeError("Expected '(' after contains")
            pop()
            # next token: field
            if not tokens or tokens[0][0] != 'ID':
                raise RuntimeError("Expected field name in contains")
            field = tokens.pop(0)[1]
            if field not in columns:
                raise RuntimeError(f"Invalid field '{field}' in contains")
            if not tokens or tokens[0][0] != 'COMMA':
                raise RuntimeError("Expected ',' after field in contains")
            pop()
            if not tokens or tokens[0][0] != 'STRING':
                raise RuntimeError("Expected string in contains")
            value_str = tokens.pop(0)[1]
            if not tokens or tokens[0][0] != 'RPAREN':
                raise RuntimeError("Expected ')' at end of contains")
            pop()
            pname = f"p{param_idx}"
            param_idx_local = param_idx
            param_idx += 1
            return f"{field} LIKE :{pname}", {pname: f"%{value_str}%"}
        elif kind == 'ID':
            field = value
            if field not in columns:
                raise RuntimeError(f"Invalid field '{field}'")
            pop()
            # next: OP
            if not tokens or tokens[0][0] != 'OP':
                raise RuntimeError("Expected operator after field")
            op = tokens.pop(0)[1]
            # next: STRING or NUMBER
            if tokens and tokens[0][0] in ('STRING', 'NUMBER'):
                val_kind, val = tokens.pop(0)
                pname = f"p{param_idx}"
                param_idx_local = param_idx
                param_idx += 1
                sql_op = {
                    'eq': '=',
                    'ne': '!=',
                    'gt': '>',
                    'ge': '>=',
                    'lt': '<',
                    'le': '<=',
                }[op]
                return f"{field} {sql_op} :{pname}", {pname: val}
            else:
                raise RuntimeError("Expected value after operator")
        else:
            raise RuntimeError(f"Unexpected token: {tok}")

    def parse_and_or(lhs_sql, lhs_params):
        nonlocal param_idx
        while tokens and tokens[0][0] in ('AND', 'OR'):
            op_kind, _ = pop()
            rhs_sql, rhs_params = parse_atom()
            if op_kind == 'AND':
                lhs_sql = f"{lhs_sql} AND {rhs_sql}"
            else:
                lhs_sql = f"{lhs_sql} OR {rhs_sql}"
            lhs_params.update(rhs_params)
        return lhs_sql, lhs_params

    def parse_expr():
        atom_sql, atom_params = parse_atom()
        return parse_and_or(atom_sql, atom_params)

    sql, params = parse_expr()
    return sql, params

def safe_parse_filter(odata_filter, columns):
    tokens = list(tokenize_filter(odata_filter))
    sql, params = parse_filter_expr(tokens, columns)
    if tokens:
        raise RuntimeError("Unexpected tokens after parsing filter")
    return sql, params

# ----------------------------------------
# Helper functions and endpoint logic
# ----------------------------------------

def parse_orderby(orderby_str, columns):
    orders = []
    for part in orderby_str.split(","):
        part = part.strip()
        if not part:
            continue
        tokens = part.split()
        field = tokens[0]
        if field not in columns:
            continue
        if len(tokens) == 2 and tokens[1].lower() in ("asc", "desc"):
            orders.append(f"{field} {tokens[1].upper()}")
        else:
            orders.append(f"{field} ASC")
    return ", ".join(orders)

def parse_odata_to_sql(query_params, columns):
    where_clause = ""
    filter_params = {}
    if "$filter" in query_params and query_params["$filter"]:
        try:
            wc, fp = safe_parse_filter(query_params["$filter"], columns)
            if wc:
                where_clause = "WHERE " + wc
                filter_params = fp
        except Exception as e:
            raise HTTPException(400, f"Error en filtro: {str(e)}")
    fields = columns
    if "$select" in query_params and query_params["$select"]:
        fields = [
            f.strip() for f in query_params["$select"].split(",")
            if f.strip() in columns
        ]
    order_clause = ""
    if "$orderby" in query_params and query_params["$orderby"]:
        sql_order = parse_orderby(query_params["$orderby"], columns)
        if sql_order:
            order_clause = "ORDER BY " + sql_order
    return fields, where_clause, order_clause, filter_params

def get_pk_columns(table_obj):
    return [col.name for col in table_obj.primary_key.columns]

def get_path_params(pk_columns):
    return "/" + "/".join([f"{{{col}}}" for col in pk_columns])

def get_pk_kwargs(pk_columns, **kwargs):
    return {col: kwargs[col] for col in pk_columns}

def get_autoinc_fields(table_obj):
    return [col.name for col in table_obj.columns if getattr(col, "autoincrement", False) or (col.primary_key and col.type.__class__.__name__ == "INTEGER")]

def get_timestamp_fields(table_obj):
    default_fields = []
    onupdate_fields = []
    for col in table_obj.columns:
        # DEFAULT CURRENT_TIMESTAMP
        if hasattr(col, "server_default") and col.server_default is not None:
            default_txt = str(col.server_default.arg).upper()
            if "CURRENT_TIMESTAMP" in default_txt:
                default_fields.append(col.name)
        extra = getattr(col, "extra", "").upper() if hasattr(col, "extra") else ""
        if "ON UPDATE" in extra or "ON UPDATE CURRENT_TIMESTAMP" in extra:
            onupdate_fields.append(col.name)
    return default_fields, onupdate_fields

def get_column_types(table_obj, pk_columns):
    import sqlalchemy
    type_map = {}
    for col in table_obj.columns:
        if col.name in pk_columns:
            if isinstance(col.type, sqlalchemy.Integer):
                type_map[col.name] = int
            elif isinstance(col.type, sqlalchemy.String):
                type_map[col.name] = str
            elif isinstance(col.type, sqlalchemy.DateTime):
                type_map[col.name] = str
            else:
                type_map[col.name] = str
    return type_map

@app.exception_handler(Exception)
async def catch_all_exceptions(request: Request, exc: Exception):
    return JSONResponse(status_code=200, content={
        "status": False,
        "value": str(exc)
    })

def make_get_all_endpoint(table_name, columns):
    async def get_all(
        filter_: str = Query(default=None, alias="$filter"),
        orderby: str = Query(default=None, alias="$orderby"),
        select: str = Query(default=None, alias="$select"),
        page: int = Query(default=1, alias="$page", ge=1),
        page_size: int = Query(default=50, alias="$pageSize", ge=1, le=1000)
    ):
        params = {}
        if filter_: params["$filter"] = filter_
        if orderby: params["$orderby"] = orderby
        if select: params["$select"] = select
        fields, where, order, filter_params = parse_odata_to_sql(params, columns)
        limit = page_size
        offset = (page - 1) * page_size
        count_query = f"SELECT COUNT(*) FROM {table_name} {where}"
        with engine.connect() as conn:
            totalCount = conn.execute(text(count_query), filter_params).scalar()
            query = f"SELECT {', '.join(fields)} FROM {table_name} {where} {order} LIMIT {limit} OFFSET {offset}"
            result = conn.execute(text(query), filter_params).fetchall()
        return {
            "status": True,
            "value": [dict(row._mapping) for row in result],
            "page": page,
            "pageSize": page_size,
            "totalCount": totalCount
        }
    return get_all

def make_get_by_pk_endpoint(table_name, pk_columns):
    async def get_by_pk(**kwargs):
        pk_kwargs = get_pk_kwargs(pk_columns, **kwargs)
        with engine.connect() as conn:
            where = " AND ".join([f"{col}=:{col}" for col in pk_columns])
            query = f"SELECT * FROM {table_name} WHERE {where}"
            result = conn.execute(text(query), pk_kwargs).first()
        if not result:
            return {"status": False, "value": "Not found"}
        return {"status": True, "value": dict(result._mapping)}
    return get_by_pk

def make_create_item_endpoint(table_name, pk_columns, columns, autoinc_fields, default_ts_fields, onupdate_ts_fields):
    async def create_item(item: dict):
        with engine.connect() as conn:
            item_clean = dict(item)
            for field in autoinc_fields:
                if field in item_clean and (item_clean[field] is None or item_clean[field] == 0 or item_clean[field] == "0"):
                    del item_clean[field]
            for field in set(default_ts_fields + onupdate_ts_fields):
                if field in item_clean:
                    del item_clean[field]
            keys = ", ".join(item_clean.keys())
            vals = ", ".join([f":{k}" for k in item_clean.keys()])
            query = f"INSERT INTO {table_name} ({keys}) VALUES ({vals})"
            result = conn.execute(text(query), item_clean)
            conn.commit()
            pk_data = {}
            for pk in pk_columns:
                if pk in autoinc_fields and (pk not in item or item[pk] is None or item[pk] == 0 or item[pk] == "0"):
                    pk_data[pk] = result.lastrowid
                else:
                    pk_data[pk] = item.get(pk)
            where = " AND ".join([f"{col}=:{col}" for col in pk_columns])
            select_query = f"SELECT * FROM {table_name} WHERE {where}"
            reg = conn.execute(text(select_query), pk_data).first()
            return {
                "status": True,
                "value": dict(reg._mapping) if reg else pk_data
            }
    return create_item

def make_update_item_endpoint(table_name, pk_columns, columns, onupdate_ts_fields):
    def endpoint_factory(pk_types):
        params_code = ", ".join([f"{col}: {pk_types[col].__name__} = Path(...)" for col in pk_columns])
        func_code = f"""
async def endpoint(item: dict = Body(...), {params_code}):
    pk_kwargs = {{{", ".join([f'"{col}": {col}' for col in pk_columns])}}}
    with engine.connect() as conn:
        item_clean = {{k: v for k, v in item.items() if k not in pk_kwargs and k not in {onupdate_ts_fields!r}}}
        if not item_clean:
            # Forzar update dummy para disparar ON UPDATE (modificación de timestamp)
            non_pk_fields = [c for c in {columns!r} if c not in pk_kwargs and c not in {onupdate_ts_fields!r}]
            if non_pk_fields:
                dummy_field = non_pk_fields[0]
                sets = f"{{dummy_field}} = {{dummy_field}}"
                params = {{**pk_kwargs}}
            else:
                return {{"status": False, "value": "No fields to update"}}
        else:
            sets = ", ".join([f"{{k}}=:{{k}}" for k in item_clean.keys()])
            params = {{**item_clean, **pk_kwargs}}
        where = " AND ".join([f"{{col}}=:{{col}}" for col in pk_kwargs.keys()])
        query = f"UPDATE {table_name} SET {{sets}} WHERE {{where}}"
        res = conn.execute(text(query), params)
        conn.commit()
        if res.rowcount == 0:
            return {{"status": False, "value": "Not found"}}
        select_query = f"SELECT * FROM {table_name} WHERE {{where}}"
        reg = conn.execute(text(select_query), pk_kwargs).first()
        return {{"status": True, "value": dict(reg._mapping) if reg else pk_kwargs}}
"""
        local_vars = {"engine": engine, "text": text, "Body": Body, "Path": Path}
        exec(func_code, local_vars)
        return local_vars["endpoint"]
    return endpoint_factory

def make_delete_item_endpoint(table_name, pk_columns, columns):
    def endpoint_factory(pk_types):
        params_code = ", ".join([f"{col}: {pk_types[col].__name__} = Path(...)" for col in pk_columns])
        func_code = f"""
async def endpoint({params_code}):
    pk_kwargs = {{{", ".join([f'"{col}": {col}' for col in pk_columns])}}}
    with engine.connect() as conn:
        where = " AND ".join([f"{{col}}=:{{col}}" for col in pk_columns])
        select_query = f"SELECT * FROM {table_name} WHERE {{where}}"
        reg = conn.execute(text(select_query), pk_kwargs).first()
        if not reg:
            return {{"status": False, "value": "Not found"}}
        reg_dict = dict(reg._mapping)
        query = f"DELETE FROM {table_name} WHERE {{where}}"
        res = conn.execute(text(query), pk_kwargs)
        conn.commit()
        return {{"status": True, "value": reg_dict}}
"""
        local_vars = {"engine": engine, "text": text, "Path": Path}
        exec(func_code, local_vars)
        return local_vars["endpoint"]
    return endpoint_factory

# --- SOLO GET para VISTAS ---
all_views = inspector.get_view_names()
for name in all_views:
    table_obj = Table(name, metadata, autoload_with=engine)
    columns = [col.name for col in table_obj.columns]
    endpoint = f"/{name}"
    app.get(endpoint)(make_get_all_endpoint(name, columns))

# --- CRUD para TABLAS ---
all_tables = inspector.get_table_names()
for name in all_tables:
    table_obj = Table(name, metadata, autoload_with=engine)
    columns = [col.name for col in table_obj.columns]
    pk_columns = get_pk_columns(table_obj)
    pk_types = get_column_types(table_obj, pk_columns)
    autoinc_fields = get_autoinc_fields(table_obj)
    default_ts_fields, onupdate_ts_fields = get_timestamp_fields(table_obj)
    endpoint = f"/{name}"

    app.get(endpoint)(make_get_all_endpoint(name, columns))

    if pk_columns:
        route = endpoint + get_path_params(pk_columns)
        # GET by PK
        def endpoint_factory_with_pk(pk_types):
            params_code = ", ".join([f"{col}: {pk_types[col].__name__} = Path(...)" for col in pk_columns])
            func_code = f"""
async def endpoint({params_code}):
    pk_kwargs = {{{", ".join([f'"{col}": {col}' for col in pk_columns])}}}
    return await handler_func(**pk_kwargs)
"""
            local_vars = {"handler_func": make_get_by_pk_endpoint(name, pk_columns), "Path": Path}
            exec(func_code, local_vars)
            return local_vars["endpoint"]
        app.get(route)(endpoint_factory_with_pk(pk_types))

        # PUT by PK (Swagger-friendly, body+pk in URL)
        app.put(route)(make_update_item_endpoint(name, pk_columns, columns, onupdate_ts_fields)(pk_types))

        # DELETE by PK
        app.delete(route)(make_delete_item_endpoint(name, pk_columns, columns)(pk_types))

        # POST
        app.post(endpoint)(make_create_item_endpoint(name, pk_columns, columns, autoinc_fields, default_ts_fields, onupdate_ts_fields))
