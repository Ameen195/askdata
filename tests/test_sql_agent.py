import pytest

from app.sql_agent import validate_sql


@pytest.mark.parametrize("sql", ["DROP TABLE orders", "INSERT INTO orders VALUES (1)"])
def test_validate_sql_rejects_mutations(sql):
    with pytest.raises(ValueError):
        validate_sql(sql)


def test_validate_sql_adds_limit():
    result = validate_sql("SELECT order_id FROM orders")
    assert "LIMIT 100" in result.upper()
