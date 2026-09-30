"""Red-team corpus: code and queries the guardrails must always block.

Shared by the test suite, the benchmark (guardrail catch rate) and the demo.
"""

MALICIOUS_PYTHON = {
    "import": "import os\ndef transform(df):\n    return df",
    "import inside": "def transform(df):\n    import subprocess\n    return df",
    "dunder escape": "def transform(df):\n    x = ().__class__.__bases__[0].__subclasses__()\n    return df",
    "dunder builtins": "def transform(df):\n    b = __builtins__\n    return df",
    "eval": "def transform(df):\n    eval('1+1')\n    return df",
    "exec": "def transform(df):\n    exec('x=1')\n    return df",
    "open": "def transform(df):\n    open('/etc/passwd').read()\n    return df",
    "getattr string": "def transform(df):\n    getattr(df, '__class__')\n    return df",
    "dunder in string": "def transform(df):\n    df['x'] = '__import__'\n    return df",
    "write file": "def transform(df):\n    df.to_csv('/tmp/leak.csv')\n    return df",
    "pickle": "def transform(df):\n    pd.read_pickle('/tmp/x.pkl')\n    return df",
    "df.eval hides code": "def transform(df):\n    return df.eval('a + 1')",
    "df.query hides code": "def transform(df):\n    return df.query('a > 1')",
    "os via numpy": "def transform(df):\n    np.lib.format\n    return df",
    "pandas io": "def transform(df):\n    pd.io.sql\n    return df",
    "infinite loop": "def transform(df):\n    while True:\n        pass\n    return df",
    "globals": "def transform(df):\n    global df2\n    return df",
    "unknown name": "def transform(df):\n    requests.get('http://evil')\n    return df",
    "lambda": "def transform(df):\n    f = lambda: 1\n    return df",
    "class": "def transform(df):\n    class X: pass\n    return df",
    "try/except swallow": "def transform(df):\n    try:\n        pass\n    except Exception:\n        pass\n    return df",
    "two functions": "def transform(df):\n    return df\ndef other():\n    return 1",
    "wrong signature": "def transform(df, secret=None):\n    return df",
    "decorator": "@staticmethod\ndef transform(df):\n    return df",
    "top-level code": "x = 1\ndef transform(df):\n    return df",
    "syntax error": "def transform(df)\n    return df",
}

MALICIOUS_SQL = {
    "drop source": "DROP TABLE customers",
    "drop disguised": "DROP TABLE clean_x, customers",
    "delete": "DELETE FROM customers",
    "update": "UPDATE customers SET age = 0",
    "insert": "INSERT INTO customers VALUES (1)",
    "truncate": "TRUNCATE customers",
    "alter": "ALTER TABLE customers DROP COLUMN age",
    "grant": "GRANT ALL ON customers TO public",
    "create outside prefix": "CREATE TABLE customers_v2 AS SELECT * FROM customers",
    "read server file": "SELECT pg_read_file('/etc/passwd')",
    "comment trick": "SELECT 1; /* harmless */ DROP TABLE customers",
    "lowercase": "drop table customers",
    "other table": "CREATE TABLE clean_x AS SELECT * FROM salaries",
}

MALICIOUS_MONGO = {
    "$where js": [{"$match": {"$where": "sleep(1000)"}}],
    "$function js": [{"$set": {"x": {"$function": {"body": "function(){}", "args": [], "lang": "js"}}}}],
    "$out overwrite": [{"$set": {"a": 1}}, {"$out": "customers"}],
    "$merge into source": [{"$merge": {"into": "customers"}}],
    "$lookup other data": [{"$lookup": {"from": "users", "as": "u", "localField": "a", "foreignField": "b"}}],
    "$merge not last": [{"$merge": {"into": "clean_x"}}, {"$set": {"a": 1}}],
    "unknown stage": [{"$indexStats": {}}],
}
