"""Bounded recursive CTE support for DPSQL.

The original DPSQL rewrite pipeline expects a flat SelectStmt.  Recursive CTEs
(`WITH RECURSIVE ...`) break that assumption because table aliases inside the
recursive term are not visible from the outer query.  This module handles the
common linear-recursive pattern separately by producing the row-level input that
DPSQL algorithms need directly:

    WITH RECURSIVE r AS (...)
    SELECT count(*) FROM r

becomes roughly:

    WITH RECURSIVE r(..., id0, ...) AS (... id columns ...)
    SELECT 1, id0, ... FROM r

The recursive term is also bounded with `--recursion-bound` when the query does
not already contain a tighter depth predicate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class IdColumn:
    table: str
    alias: str
    pk: str
    name: str


def is_recursive_query(query: str) -> bool:
    return bool(re.search(r"\bWITH\s+RECURSIVE\b", query, flags=re.IGNORECASE))


def _strip_semicolon(query: str) -> str:
    return query.strip().rstrip(";").strip()


def _split_cte(query: str) -> Optional[Tuple[str, List[str], str, str]]:
    """Return (cte_name, column_names, cte_body, outer_select) for one CTE.

    This intentionally supports the project use case: a single recursive CTE
    followed by a final SELECT.  It is conservative and raises a helpful error
    through the caller for unsupported shapes instead of silently producing
    invalid SQL.
    """
    q = _strip_semicolon(query)
    m = re.match(
        r"\s*WITH\s+RECURSIVE\s+(?P<name>[A-Za-z_][\w]*)\s*(?P<cols>\([^)]*\))?\s+AS\s*\(",
        q,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return None

    open_idx = m.end() - 1
    depth = 0
    close_idx = None
    for i in range(open_idx, len(q)):
        ch = q[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                close_idx = i
                break
    if close_idx is None:
        return None

    cols_raw = m.group("cols")
    cols = []
    if cols_raw:
        cols = [c.strip() for c in cols_raw[1:-1].split(",") if c.strip()]

    return m.group("name"), cols, q[open_idx + 1 : close_idx].strip(), q[close_idx + 1 :].strip()


def _split_union_all(cte_body: str) -> Tuple[str, str]:
    # Split at top-level UNION ALL only.
    depth = 0
    upper = cte_body.upper()
    i = 0
    while i < len(cte_body):
        ch = cte_body[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and upper.startswith("UNION ALL", i):
            return cte_body[:i].strip(), cte_body[i + len("UNION ALL") :].strip()
        i += 1
    raise ValueError("Recursive CTE must contain a top-level UNION ALL.")


def _pk_lookup(pks: Sequence[Sequence[str]], private_relations: str) -> dict:
    relation_set = {r.strip() for r in private_relations.split(",") if r.strip()}
    out = {}
    for pk in pks:
        table = str(pk[0])
        if table not in relation_set:
            continue
        definition = str(pk[2])
        left = definition.find("(")
        right = definition.find(")")
        if left != -1 and right != -1 and right > left:
            out[table] = definition[left + 1 : right].strip()
    return out


def _infer_pk_from_recursive_sql(table: str, anchor_sql: str, recursive_sql: str) -> Optional[str]:
    """Best-effort fallback when PostgreSQL primary key metadata is absent.

    DPSQL needs a stable tuple identifier for each private tuple contribution.
    Some demo graph tables do not declare a DB primary key, so primary_keys.txt
    returns nothing. For bounded graph recursion, the edge tuple is usually
    identified by the columns used as alias.column in the anchor/recursive terms,
    e.g. graph_edges(src, dst).
    """
    cols: List[str] = []
    combined = anchor_sql + "\n" + recursive_sql
    pat = re.compile(
        rf"\b(?:FROM|JOIN)\s+{re.escape(table)}(?:\s+(?:AS\s+)?([A-Za-z_]\w*))?",
        flags=re.IGNORECASE,
    )
    aliases = []
    for m in pat.finditer(combined):
        alias = m.group(1) or table
        if alias.upper() in {"WHERE", "JOIN", "INNER", "LEFT", "RIGHT", "FULL", "CROSS", "ON", "GROUP", "ORDER", "UNION"}:
            alias = table
        aliases.append(alias)

    for alias in aliases:
        for cm in re.finditer(rf"\b{re.escape(alias)}\.([A-Za-z_]\w*)\b", combined):
            col = cm.group(1)
            if col not in cols:
                cols.append(col)

    # Prefer the natural edge key when present.
    lowered = {c.lower(): c for c in cols}
    if "src" in lowered and "dst" in lowered:
        return f"{lowered['src']}, {lowered['dst']}"
    if cols:
        return ", ".join(cols[:2])
    return None


def _fill_missing_pk_fallbacks(
    pk_by_table: dict,
    private_relations: str,
    anchor_sql: str,
    recursive_sql: str,
) -> dict:
    """Fill missing private relation keys using recursive SQL as fallback."""
    out = dict(pk_by_table)
    for table in [r.strip() for r in private_relations.split(",") if r.strip()]:
        if table in out:
            continue
        inferred = _infer_pk_from_recursive_sql(table, anchor_sql, recursive_sql)
        if inferred:
            out[table] = inferred
    return out


def _qualified_pk_expr(alias: str, pk: str) -> str:
    """Return a SQL expression identifying a private tuple.

    Supports both single-column and composite primary keys:
    - Single:   _qualified_pk_expr("e", "id")       → "e.id"
    - Composite: _qualified_pk_expr("e", "src,dst") → "concat(e.src, ':', e.dst)"
    
    Composite keys are concatenated with ':' as separator to produce
    a single stable tuple identifier for differential privacy.
    """
    cols = [c.strip() for c in pk.split(",") if c.strip()]
    if not cols:
        raise ValueError("Primary key definition did not contain any columns.")
    
    # Single column: return qualified name directly
    if len(cols) == 1:
        return f"{alias}.{cols[0]}"
    
    # Composite key: concat with ':' separator
    qualified = [f"{alias}.{col}" for col in cols]
    parts = [qualified[0]]
    for col in qualified[1:]:
        parts.append("':'")  # String literal ':' to be concatenated
        parts.append(col)
    return "concat(" + ", ".join(parts) + ")"


def _find_private_aliases(sql: str, pk_by_table: dict, cte_name: str, start_index: int = 0) -> List[IdColumn]:
    found: List[IdColumn] = []
    idx = start_index
    for table, pk in pk_by_table.items():
        pat = re.compile(
            rf"\b(?:FROM|JOIN)\s+{re.escape(table)}(?:\s+(?:AS\s+)?([A-Za-z_]\w*))?",
            flags=re.IGNORECASE,
        )
        for m in pat.finditer(sql):
            alias = m.group(1) or table
            if alias.upper() in {"WHERE", "JOIN", "INNER", "LEFT", "RIGHT", "FULL", "CROSS", "ON", "GROUP", "ORDER", "UNION"}:
                alias = table
            if alias.lower() == cte_name.lower():
                continue
            found.append(IdColumn(table=table, alias=alias, pk=pk, name=f"id{idx}"))
            idx += 1
    return found


def _insert_targets(select_sql: str, targets: Iterable[str]) -> str:
    targets = list(targets)
    if not targets:
        return select_sql
    m = re.search(r"\bFROM\b", select_sql, flags=re.IGNORECASE)
    if not m:
        raise ValueError("Could not add DPSQL id columns: SELECT term has no FROM clause.")
    return select_sql[: m.start()].rstrip() + ", " + ", ".join(targets) + " " + select_sql[m.start() :].lstrip()


def _depth_alias_and_column(recursive_sql: str, cte_cols: Sequence[str], cte_name: str) -> Tuple[Optional[str], Optional[str]]:
    # Prefer an explicit alias in patterns such as r.depth + 1.
    m = re.search(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\s*\+\s*1\b", recursive_sql, flags=re.IGNORECASE)
    if m and m.group(1).lower() != cte_name.lower():
        return m.group(1), m.group(2)
    # Fall back to a common CTE column name.
    for c in cte_cols:
        if c.lower() in {"depth", "level", "hop", "hops"}:
            return None, c
    return None, None


def _apply_recursion_bound(recursive_sql: str, cte_cols: Sequence[str], cte_name: str, bound: int) -> str:
    if bound < 1:
        raise ValueError("recursion_bound must be >= 1.")
    alias, depth_col = _depth_alias_and_column(recursive_sql, cte_cols, cte_name)
    if not depth_col:
        # The query may already be naturally bounded by another predicate.  Keep it unchanged.
        return recursive_sql
    depth_ref = f"{alias}.{depth_col}" if alias else depth_col
    predicate = f"{depth_ref} < {int(bound)}"

    # Always append the CLI bound.  If the query already has a stricter bound,
    # this is redundant; if it has a looser bound, this safely tightens it.
    if re.search(r"\bWHERE\b", recursive_sql, flags=re.IGNORECASE):
        return recursive_sql.rstrip() + f" AND {predicate}"
    return recursive_sql.rstrip() + f" WHERE {predicate}"


def _outer_select_to_row_input(outer_select: str, cte_name: str, id_columns: Sequence[IdColumn]) -> str:
    ids = ", ".join(c.name for c in id_columns)
    suffix = f", {ids}" if ids else ""
    # COUNT/SUM/MAX become row-level rows consumed by DPSQL algorithms.
    m_count = re.match(rf"\s*SELECT\s+count\s*\(\s*\*\s*\)\s+FROM\s+{re.escape(cte_name)}\b.*", outer_select, flags=re.IGNORECASE | re.DOTALL)
    if m_count:
        return f"SELECT 1{suffix} FROM {cte_name}"

    m_sum = re.match(rf"\s*SELECT\s+sum\s*\((?P<expr>.*?)\)\s+FROM\s+{re.escape(cte_name)}\b.*", outer_select, flags=re.IGNORECASE | re.DOTALL)
    if m_sum:
        return f"SELECT {m_sum.group('expr')}{suffix} FROM {cte_name}"

    # Non-aggregate final select: append id columns so the DP layer still has user ids.
    return _insert_targets(outer_select, [c.name for c in id_columns])


def rewrite_bounded_recursive_query(
    query: str,
    private_relations: str,
    pks: Sequence[Sequence[str]],
    recursion_bound: int,
) -> Optional[str]:
    """Rewrite a bounded linear recursive CTE into DPSQL row-level input.

    Returns None when the query is not recursive.  Raises ValueError for a
    recursive query shape that is not supported by this project extension.
    """
    if not is_recursive_query(query):
        return None

    split = _split_cte(query)
    if split is None:
        raise ValueError("Only single WITH RECURSIVE cte AS (...) SELECT ... queries are supported.")

    cte_name, cte_cols, cte_body, outer_select = split
    anchor_sql, recursive_sql = _split_union_all(cte_body)
    pk_by_table = _pk_lookup(pks, private_relations)
    pk_by_table = _fill_missing_pk_fallbacks(pk_by_table, private_relations, anchor_sql, recursive_sql)
    if not pk_by_table:
        raise ValueError(
            "No primary key was found or inferred for the private recursive relation. "
            "Either declare a DB primary key or use edge columns such as src/dst in the recursive SQL."
        )

    anchor_aliases = _find_private_aliases(anchor_sql, pk_by_table, cte_name, 0)
    recursive_aliases = _find_private_aliases(recursive_sql, pk_by_table, cte_name, 0)
    if not anchor_aliases or not recursive_aliases:
        raise ValueError("Could not find private table references in both anchor and recursive CTE terms.")
    if len(anchor_aliases) != 1 or len(recursive_aliases) != 1:
        raise ValueError("Bounded recursion currently supports one private table reference per anchor/recursive term.")

    anchor_edge = anchor_aliases[0]
    recursive_edge = recursive_aliases[0]
    recursive_cte_alias, depth_col = _depth_alias_and_column(recursive_sql, cte_cols, cte_name)
    if not recursive_cte_alias or not depth_col:
        raise ValueError("Could not infer recursive alias/depth column. Use a pattern such as r.depth + 1.")

    # One id column per possible hop.  This preserves the full path contribution:
    # depth 1 -> id0, depth 2 -> id0,id1, ..., depth B -> id0..id(B-1).
    id_names = [f"id{i}" for i in range(int(recursion_bound))]
    anchor_targets = [
        f"concat('id0', {_qualified_pk_expr(anchor_edge.alias, anchor_edge.pk)}) AS id0",
        *[f"NULL AS {name}" for name in id_names[1:]],
    ]
    recursive_targets = [f"{recursive_cte_alias}.id0 AS id0"]
    current_id_expr = f"concat('id', {recursive_cte_alias}.{depth_col}, {_qualified_pk_expr(recursive_edge.alias, recursive_edge.pk)})"
    for i, name in enumerate(id_names[1:], start=1):
        recursive_targets.append(
            f"CASE WHEN {recursive_cte_alias}.{depth_col} = {i} "
            f"THEN {current_id_expr} ELSE {recursive_cte_alias}.{name} END AS {name}"
        )

    bounded_recursive_sql = _apply_recursion_bound(recursive_sql, cte_cols, cte_name, recursion_bound)
    anchor_sql = _insert_targets(anchor_sql, anchor_targets)
    bounded_recursive_sql = _insert_targets(bounded_recursive_sql, recursive_targets)

    cte_column_suffix = ""
    if cte_cols:
        cte_column_suffix = "(" + ", ".join(list(cte_cols) + id_names) + ")"

    row_input_select = _outer_select_to_row_input(
        outer_select,
        cte_name,
        [IdColumn(table=anchor_edge.table, alias=cte_name, pk="", name=name) for name in id_names],
    )
    return (
        f"WITH RECURSIVE {cte_name}{cte_column_suffix} AS (\n"
        f"  {anchor_sql}\n"
        f"  UNION ALL\n"
        f"  {bounded_recursive_sql}\n"
        f")\n{row_input_select}"
    )
