import psycopg2
try:
    conn = psycopg2.connect(database="tpch")
    print("Connected to 'tpch' without credentials!")
    conn.close()
except Exception as e:
    print(f"Failed to connect to 'tpch': {e}")

try:
    conn = psycopg2.connect(database="postgres")
    print("Connected to 'postgres' without credentials!")
    conn.close()
except Exception as e:
    print(f"Failed to connect to 'postgres': {e}")
