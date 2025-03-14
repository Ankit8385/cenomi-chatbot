import psycopg2
from config import DB_POOL

def fetch_schema() -> str:
    """Fetches database schema."""
    conn = None
    try:
        conn = DB_POOL.getconn()
        if not conn:
            return "Database connection failed"

        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name, column_name, data_type 
                FROM information_schema.columns 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            columns = cur.fetchall()

            schema_dict = {}
            for table, column, dtype in columns:
                if table not in schema_dict:
                    schema_dict[table] = []
                schema_dict[table].append(f"{column} ({dtype})")

            schema = "Dialect: PostgreSQL\n" + "\n".join(
                f"{table}: {', '.join(cols)}" 
                for table, cols in schema_dict.items()
            )
            return schema

    except psycopg2.Error as e:
        return f"Schema error: {str(e)}"
    finally:
        if conn:
            DB_POOL.putconn(conn)

# Set schema as a global variable for nodes to use
schema = fetch_schema()