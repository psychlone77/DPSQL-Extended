"""
Unit tests for recursive CTE unrolling in DPSQL-Extended.

Tests the apply_recursive_unroll function which transforms
WITH RECURSIVE queries into flat SELECT statements with
bounded self-joins.
"""

import sys
import os
import warnings

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from pglast import parser, stream, ast
from src.recursive_cte import apply_recursive_unroll


def _unroll(sql, k=3):
    """Helper: parse SQL, unroll, and return the raw SQL string."""
    root = parser.parse_sql(sql)
    stmt = root[0].stmt
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = apply_recursive_unroll(stmt, k=k)
    return stream.RawStream()(result)


# ── Detection Tests ──────────────────────────────────────────────────

class TestCTEDetection:

    def test_detects_recursive_cte(self):
        sql = (
            "WITH RECURSIVE sub AS ("
            "  SELECT id FROM t WHERE pid = 1"
            "  UNION ALL"
            "  SELECT t2.id FROM t AS t2 JOIN sub s ON s.id = t2.pid"
            ") SELECT count(*) FROM sub"
        )
        result = _unroll(sql, k=2)
        # Should NOT contain WITH RECURSIVE in output
        assert "WITH" not in result.upper()
        assert "RECURSIVE" not in result.upper()

    def test_non_recursive_passthrough(self):
        sql = "SELECT count(*) FROM supplier, lineitem WHERE supplier.s_suppkey = lineitem.l_suppkey"
        result = _unroll(sql, k=3)
        # Should pass through unchanged
        assert "supplier" in result
        assert "lineitem" in result

    def test_non_recursive_cte_passthrough(self):
        """Non-recursive WITH clause (no RECURSIVE keyword) should pass through."""
        sql = (
            "WITH recent AS ("
            "  SELECT id, name FROM employees WHERE hire_date > '2020-01-01'"
            ") SELECT count(*) FROM recent"
        )
        result = _unroll(sql, k=3)
        # Should still contain WITH since it's not recursive
        assert "recent" in result.lower()


# ── Unrolling Correctness Tests ──────────────────────────────────────

class TestUnrollingCorrectness:

    RECURSIVE_SQL = (
        "WITH RECURSIVE subordinates AS ("
        "  SELECT emp_id FROM Employees WHERE manager_id = 1"
        "  UNION ALL"
        "  SELECT e.emp_id FROM Employees e"
        "  JOIN subordinates s ON s.emp_id = e.manager_id"
        ") SELECT COUNT(*) FROM subordinates"
    )

    def test_k1_no_joins(self):
        result = _unroll(self.RECURSIVE_SQL, k=1)
        assert "JOIN" not in result.upper()
        assert "e1" in result
        assert "e1.manager_id" in result

    def test_k3_two_joins(self):
        result = _unroll(self.RECURSIVE_SQL, k=3)
        upper = result.upper()
        assert upper.count("LEFT JOIN") == 2
        assert "e1" in result
        assert "e2" in result
        assert "e3" in result

    def test_k5_four_joins(self):
        result = _unroll(self.RECURSIVE_SQL, k=5)
        upper = result.upper()
        assert upper.count("LEFT JOIN") == 4
        for i in range(1, 6):
            assert f"e{i}" in result

    def test_base_filter_applied_to_first_alias(self):
        result = _unroll(self.RECURSIVE_SQL, k=3)
        assert "e1.manager_id = 1" in result

    def test_join_chain_connects_aliases(self):
        result = _unroll(self.RECURSIVE_SQL, k=3)
        assert "e1.emp_id = e2.manager_id" in result
        assert "e2.emp_id = e3.manager_id" in result

    def test_preserves_outer_aggregation(self):
        result = _unroll(self.RECURSIVE_SQL, k=3)
        assert "count(*)" in result.lower()

    def test_with_clause_removed(self):
        """After unrolling, the withClause should be None."""
        root = parser.parse_sql(self.RECURSIVE_SQL)
        stmt = root[0].stmt
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            apply_recursive_unroll(stmt, k=3)
        assert stmt.withClause is None


# ── Edge Cases ───────────────────────────────────────────────────────

class TestEdgeCases:

    def test_warning_emitted(self):
        sql = (
            "WITH RECURSIVE sub AS ("
            "  SELECT id FROM t WHERE pid = 1"
            "  UNION ALL"
            "  SELECT t2.id FROM t AS t2 JOIN sub s ON s.id = t2.pid"
            ") SELECT count(*) FROM sub"
        )
        root = parser.parse_sql(sql)
        stmt = root[0].stmt
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            apply_recursive_unroll(stmt, k=3)
            assert len(w) == 1
            assert "unrolled to depth k=3" in str(w[0].message)

    def test_output_is_valid_sql(self):
        """The unrolled output should be parseable by pglast."""
        sql = (
            "WITH RECURSIVE subordinates AS ("
            "  SELECT emp_id FROM Employees WHERE manager_id = 1"
            "  UNION ALL"
            "  SELECT e.emp_id FROM Employees e"
            "  JOIN subordinates s ON s.emp_id = e.manager_id"
            ") SELECT COUNT(*) FROM subordinates"
        )
        result = _unroll(sql, k=3)
        # Should not raise
        reparsed = parser.parse_sql(result)
        assert reparsed is not None

    def test_sum_aggregation_preserved(self):
        sql = (
            "WITH RECURSIVE sub AS ("
            "  SELECT id, salary FROM t WHERE pid = 1"
            "  UNION ALL"
            "  SELECT t2.id, t2.salary FROM t AS t2 JOIN sub s ON s.id = t2.pid"
            ") SELECT SUM(salary) FROM sub"
        )
        result = _unroll(sql, k=2)
        assert "sum" in result.lower()
        assert "WITH" not in result.upper()

# ── Pipeline Integration Tests ───────────────────────────────────────

class TestPipelineIntegration:
    """Test that LEFT JOINs from unrolling survive the downstream pipeline."""

    RECURSIVE_SQL = (
        "WITH RECURSIVE subordinates AS ("
        "  SELECT emp_id FROM Employees WHERE manager_id = 1"
        "  UNION ALL"
        "  SELECT e.emp_id FROM Employees e"
        "  JOIN subordinates s ON s.emp_id = e.manager_id"
        ") SELECT COUNT(*) FROM subordinates"
    )

    def _unroll_stmt(self, sql, k=3):
        root = parser.parse_sql(sql)
        stmt = root[0].stmt
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return apply_recursive_unroll(stmt, k=k)

    def test_implicit_join_preserves_left_joins(self):
        """ImplicitJoin should NOT decompose LEFT JOINs."""
        from src.parser import ImplicitJoin
        stmt = self._unroll_stmt(self.RECURSIVE_SQL, k=3)
        ImplicitJoin()(stmt)
        result = stream.RawStream()(stmt)
        assert "LEFT JOIN" in result.upper()
        assert result.upper().count("LEFT JOIN") == 2

    def test_implicit_join_still_decomposes_inner_joins(self):
        """ImplicitJoin should still work on regular INNER JOIN queries."""
        from src.parser import ImplicitJoin
        sql = "SELECT * FROM a INNER JOIN b ON a.id = b.aid"
        root = parser.parse_sql(sql)
        stmt = root[0].stmt
        ImplicitJoin()(stmt)
        result = stream.RawStream()(stmt)
        # INNER JOIN should be decomposed into comma + WHERE
        assert "JOIN" not in result.upper()
        assert "WHERE" in result.upper()

    def test_get_rename_finds_tables_in_left_joins(self):
        """get_rename should discover all table aliases inside JoinExpr trees."""
        from src.parser import get_rename
        stmt = self._unroll_stmt(self.RECURSIVE_SQL, k=3)
        renaming = get_rename()
        renaming(stmt)
        assert "employees" in renaming.rename_dict
        aliases = renaming.rename_dict["employees"]
        assert "e1" in aliases
        assert "e2" in aliases
        assert "e3" in aliases
        assert len(aliases) == 3

    def test_full_pipeline_no_crash(self):
        """The full rewrite pipeline should not crash on recursive CTEs."""
        from src.parser import ImplicitJoin, get_rename, add_table_name, aggregationVisit
        stmt = self._unroll_stmt(self.RECURSIVE_SQL, k=3)
        # These should all succeed without raising
        ImplicitJoin()(stmt)
        schema = {"employees": ["emp_id", "manager_id", "name"]}
        add_table_name(stmt, schema)(stmt)
        aggregationVisit()(stmt)
        result = stream.RawStream()(stmt)
        assert result  # non-empty output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
