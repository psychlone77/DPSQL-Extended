"""
Recursive CTE Unrolling for DPSQL-Extended.

Transforms recursive CTEs (WITH RECURSIVE) into bounded self-join queries
by unrolling the recursion to a configurable depth k. This enables the
existing differential privacy pipeline to compute sensitivity and apply noise,
since unbounded recursion makes sensitivity analysis impossible.

Example transformation (k=3):

    WITH RECURSIVE subordinates AS (
        SELECT emp_id FROM Employees WHERE manager_id = 1
        UNION ALL
        SELECT e.emp_id FROM Employees e
        JOIN subordinates s ON s.emp_id = e.manager_id
    )
    SELECT COUNT(*) FROM subordinates;

    =>

    SELECT count(*)
    FROM employees AS e1
    LEFT JOIN employees AS e2 ON e1.emp_id = e2.manager_id
    LEFT JOIN employees AS e3 ON e2.emp_id = e3.manager_id
    WHERE e1.manager_id = 1
"""

import copy
import warnings

from pglast import ast, enums
from pglast.visitors import Visitor


def _find_cte_reference(from_clause, cte_name):
    """
    Walk the FROM clause to find the RangeVar referencing the CTE name.
    Returns (real_table_rangevar, cte_rangevar, join_quals) or raises if not found.

    The recursive step typically looks like:
        FROM real_table AS alias JOIN cte_name AS alias2 ON ...
    """
    for item in from_clause:
        if isinstance(item, ast.JoinExpr):
            larg = item.larg
            rarg = item.rarg
            # Case 1: LEFT is real table, RIGHT is CTE reference
            if isinstance(rarg, ast.RangeVar) and rarg.relname == cte_name:
                return larg, rarg, item.quals
            # Case 2: LEFT is CTE reference, RIGHT is real table
            if isinstance(larg, ast.RangeVar) and larg.relname == cte_name:
                return rarg, larg, item.quals
        elif isinstance(item, ast.RangeVar) and item.relname == cte_name:
            # CTE referenced directly in FROM (join condition is in WHERE)
            # We need to find the real table separately
            pass

    # Fallback: CTE is referenced as a standalone table in FROM
    # (join condition would be in WHERE clause)
    real_table = None
    cte_ref = None
    for item in from_clause:
        if isinstance(item, ast.RangeVar):
            if item.relname == cte_name:
                cte_ref = item
            else:
                real_table = item
    if real_table and cte_ref:
        return real_table, cte_ref, None

    raise ValueError(
        f"Could not find CTE reference '{cte_name}' in recursive step FROM clause"
    )


def _extract_join_columns(quals, cte_alias, table_alias):
    """
    Extract the column pair from a join condition like:
        cte_alias.col1 = table_alias.col2

    Returns (cte_column, table_column) — the column names used in the
    recursive join relationship.
    """
    if not isinstance(quals, ast.A_Expr):
        raise ValueError("Expected a simple A_Expr join condition in recursive step")

    if quals.kind != enums.A_Expr_Kind.AEXPR_OP:
        raise ValueError("Expected an equality operator in recursive join condition")

    lexpr = quals.lexpr
    rexpr = quals.rexpr

    if not (isinstance(lexpr, ast.ColumnRef) and isinstance(rexpr, ast.ColumnRef)):
        raise ValueError("Expected column references on both sides of recursive join")

    # Extract table.column from each side
    def _get_table_col(col_ref):
        fields = col_ref.fields
        if len(fields) == 2:
            return fields[0].sval, fields[1].sval
        elif len(fields) == 1:
            return None, fields[0].sval
        raise ValueError(f"Unexpected column reference format: {fields}")

    l_table, l_col = _get_table_col(lexpr)
    r_table, r_col = _get_table_col(rexpr)

    # Determine which side references the CTE and which references the real table
    if l_table == cte_alias:
        return l_col, r_col  # (cte_column, table_column)
    elif r_table == cte_alias:
        return r_col, l_col  # (cte_column, table_column)
    elif l_table == table_alias:
        return r_col, l_col
    elif r_table == table_alias:
        return l_col, r_col
    else:
        # No explicit table qualifier — try positional: left is CTE, right is table
        return l_col, r_col


def _remap_column_refs(node, old_alias, new_alias):
    """
    Recursively walk an AST node and replace all ColumnRef references
    from old_alias.col to new_alias.col.
    """
    if isinstance(node, ast.ColumnRef):
        fields = node.fields
        if len(fields) == 2 and isinstance(fields[0], ast.String):
            if fields[0].sval == old_alias:
                node.fields = (ast.String(sval=new_alias), fields[1])
    elif isinstance(node, ast.Node):
        for member in node:
            value = getattr(node, member)
            if isinstance(value, ast.Node):
                _remap_column_refs(value, old_alias, new_alias)
            elif isinstance(value, tuple):
                for item in value:
                    if isinstance(item, ast.Node):
                        _remap_column_refs(item, old_alias, new_alias)


def _extract_where_join_condition(where_clause, cte_alias, table_alias):
    """
    When the recursive step uses implicit join syntax (comma-separated FROM),
    extract the join condition from the WHERE clause.
    Returns (join_quals, remaining_where) where join_quals is the condition
    linking the CTE to the real table.
    """
    if where_clause is None:
        return None, None

    if isinstance(where_clause, ast.A_Expr):
        # Check if this expression references both cte and table
        return where_clause, None

    if isinstance(where_clause, ast.BoolExpr):
        if where_clause.boolop == enums.BoolExprType.AND_EXPR:
            join_parts = []
            other_parts = []
            for arg in where_clause.args:
                if isinstance(arg, ast.A_Expr):
                    # Simple heuristic: if it references the CTE alias, it's a join cond
                    join_parts.append(arg)
                else:
                    other_parts.append(arg)
            if join_parts:
                join_qual = join_parts[0] if len(join_parts) == 1 else None
                remaining = None
                if other_parts:
                    if len(other_parts) == 1:
                        remaining = other_parts[0]
                    else:
                        remaining = ast.BoolExpr(
                            boolop=enums.BoolExprType.AND_EXPR,
                            args=tuple(other_parts),
                        )
                return join_qual, remaining

    return None, where_clause


def apply_recursive_unroll(root_stmt, k=3):
    """
    Transform a SelectStmt containing a recursive CTE into a flat SelectStmt
    with k levels of explicit LEFT JOINs.

    Args:
        root_stmt: A pglast ast.SelectStmt node (the top-level query).
        k: Maximum recursion depth to unroll to.

    Returns:
        The transformed SelectStmt (withClause removed, self-joins added).
        If no recursive CTE is found, returns the statement unchanged.
    """
    # Check if there is a WITH RECURSIVE clause
    if root_stmt.withClause is None or not root_stmt.withClause.recursive:
        return root_stmt

    with_clause = root_stmt.withClause
    ctes = with_clause.ctes

    # Process each recursive CTE (typically there's only one)
    # Note: In pglast, the recursive flag is on withClause.recursive,
    # not on individual CTEs. We detect recursion by checking if the
    # CTE's query is a UNION that references the CTE name in its right arm.
    for cte in ctes:

        cte_name = cte.ctename
        cte_query = cte.ctequery

        # The CTE query should be a UNION [ALL] of base case and recursive step
        if cte_query.op != enums.SetOperation.SETOP_UNION:
            warnings.warn(
                f"Recursive CTE '{cte_name}' does not use UNION; skipping unroll."
            )
            continue

        is_union_all = cte_query.all
        base_case = cte_query.larg   # Base case SELECT
        recursive_step = cte_query.rarg  # Recursive step SELECT

        # --- Extract components from the recursive step ---
        real_table, cte_ref, join_quals = _find_cte_reference(
            recursive_step.fromClause, cte_name
        )

        # Get aliases
        real_table_name = real_table.relname
        table_alias = (
            real_table.alias.aliasname if real_table.alias else real_table_name
        )
        cte_alias = cte_ref.alias.aliasname if cte_ref.alias else cte_name

        # If join condition was in WHERE instead of ON
        if join_quals is None:
            join_quals, _remaining_where = _extract_where_join_condition(
                recursive_step.whereClause, cte_alias, table_alias
            )

        if join_quals is None:
            raise ValueError(
                f"Could not extract join condition for recursive CTE '{cte_name}'"
            )

        # Extract the column pair from the join condition
        cte_col, table_col = _extract_join_columns(
            join_quals, cte_alias, table_alias
        )

        # --- Extract the base case filter ---
        base_where = copy.deepcopy(base_case.whereClause)

        # --- Build the unrolled query ---

        # Generate aliases: e1, e2, ..., ek
        alias_prefix = real_table_name[0].lower()
        aliases = [f"{alias_prefix}{i}" for i in range(1, k + 1)]

        # First table in FROM clause (the base level)
        first_table = ast.RangeVar(
            relname=real_table_name,
            inh=True,
            relpersistence="p",
            alias=ast.Alias(aliasname=aliases[0]),
        )

        # Build the chain of LEFT JOINs
        current_from = first_table
        for i in range(1, k):
            next_table = ast.RangeVar(
                relname=real_table_name,
                inh=True,
                relpersistence="p",
                alias=ast.Alias(aliasname=aliases[i]),
            )
            # Join condition: prev_alias.cte_col = curr_alias.table_col
            join_condition = ast.A_Expr(
                kind=enums.A_Expr_Kind.AEXPR_OP,
                name=(ast.String(sval="="),),
                lexpr=ast.ColumnRef(
                    fields=(
                        ast.String(sval=aliases[i - 1]),
                        ast.String(sval=cte_col),
                    )
                ),
                rexpr=ast.ColumnRef(
                    fields=(
                        ast.String(sval=aliases[i]),
                        ast.String(sval=table_col),
                    )
                ),
            )
            current_from = ast.JoinExpr(
                jointype=enums.JoinType.JOIN_LEFT,
                isNatural=False,
                larg=current_from,
                rarg=next_table,
                quals=join_condition,
            )

        # Remap base case WHERE clause to use the first alias
        if base_where is not None:
            # The base case may reference unqualified columns — qualify them
            # with the first alias
            _qualify_bare_columns(base_where, aliases[0])

        # --- Build the new outer SELECT ---

        # Preserve the outer query's target list (e.g., COUNT(*), SUM(...))
        outer_target = root_stmt.targetList

        # Build the new SelectStmt
        new_stmt = ast.SelectStmt(
            targetList=outer_target,
            fromClause=(current_from,),
            whereClause=base_where,
            groupClause=root_stmt.groupClause,
            havingClause=root_stmt.havingClause,
            sortClause=root_stmt.sortClause,
            limitCount=root_stmt.limitCount,
            limitOffset=root_stmt.limitOffset,
        )

        # If UNION (not ALL), wrap target in DISTINCT
        if not is_union_all:
            # Add DISTINCT to the outer query
            new_stmt.distinctClause = (ast.Node(),)

        # Replace the root statement fields in-place
        root_stmt.targetList = new_stmt.targetList
        root_stmt.fromClause = new_stmt.fromClause
        root_stmt.whereClause = new_stmt.whereClause
        root_stmt.groupClause = new_stmt.groupClause
        root_stmt.havingClause = new_stmt.havingClause
        root_stmt.sortClause = new_stmt.sortClause
        root_stmt.limitCount = new_stmt.limitCount
        root_stmt.limitOffset = new_stmt.limitOffset
        root_stmt.withClause = None

        if not is_union_all:
            root_stmt.distinctClause = new_stmt.distinctClause

        warnings.warn(
            f"Recursive CTE '{cte_name}' unrolled to depth k={k}. "
            f"Results deeper than {k} levels will be truncated."
        )

    return root_stmt


def _qualify_bare_columns(node, alias):
    """
    Walk an AST node and qualify any bare (unqualified) ColumnRef nodes
    with the given table alias. e.g., ColumnRef('pid') -> ColumnRef('e1', 'pid')
    """
    if isinstance(node, ast.ColumnRef):
        fields = node.fields
        if len(fields) == 1 and isinstance(fields[0], ast.String):
            # Bare column name — qualify with alias
            node.fields = (ast.String(sval=alias), fields[0])
    elif isinstance(node, ast.Node):
        for member in node:
            value = getattr(node, member)
            if isinstance(value, ast.Node):
                _qualify_bare_columns(value, alias)
            elif isinstance(value, tuple):
                for item in value:
                    if isinstance(item, ast.Node):
                        _qualify_bare_columns(item, alias)
