import sqlite3
import re
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "northwind.sqlite"

class SQLiteTool:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self._schema_cache: Optional[Dict[str, List[Dict[str, Any]]]] = None
    
    def get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))
    
    def get_schema(self) -> Dict[str, List[Dict[str, Any]]]:
        if self._schema_cache is not None:
            return self._schema_cache
        schema = {}
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = cursor.fetchall()
            for (table_name,) in tables:
                cursor.execute(f'PRAGMA table_info("{table_name}")')
                columns = cursor.fetchall()
                schema[table_name] = [{"cid": col[0], "name": col[1], "type": col[2], "notnull": col[3], "default": col[4], "pk": col[5]} for col in columns]
        finally:
            conn.close()
        self._schema_cache = schema
        return schema
    
    def get_schema_string(self) -> str:
        schema = self.get_schema()
        lines = []
        for table_name, columns in schema.items():
            col_strs = [f"{col['name']} {col['type']}{' PK' if col['pk'] else ''}" for col in columns]
            lines.append(f'{table_name}({", ".join(col_strs)})')
        return "\n".join(lines)
    
    def execute_query(self, sql: str) -> Tuple[List[str], List[Tuple], Optional[str]]:
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(sql)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            return columns, rows, None
        except Exception as e:
            return [], [], str(e)
        finally:
            conn.close()
    
    def extract_tables_from_sql(self, sql: str) -> List[str]:
        schema = self.get_schema()
        table_names = set(schema.keys())
        found_tables = []
        for table in table_names:
            patterns = [rf'\b{re.escape(table)}\b', rf'"{re.escape(table)}"', rf"'{re.escape(table)}'", rf'\[{re.escape(table)}\]']
            for pattern in patterns:
                if re.search(pattern, sql, re.IGNORECASE):
                    if table not in found_tables:
                        found_tables.append(table)
                    break
        return found_tables
    
    def get_sample_rows(self, table_name: str, limit: int = 3) -> List[Dict[str, Any]]:
        sql = f'SELECT * FROM "{table_name}" LIMIT {limit}'
        columns, rows, error = self.execute_query(sql)
        if error:
            return []
        return [dict(zip(columns, row)) for row in rows]

_tool_instance: Optional[SQLiteTool] = None

def get_sqlite_tool(db_path: Optional[Path] = None) -> SQLiteTool:
    global _tool_instance
    if _tool_instance is None or (db_path and db_path != _tool_instance.db_path):
        _tool_instance = SQLiteTool(db_path)
    return _tool_instance

if __name__ == "__main__":
    tool = get_sqlite_tool()
    print("=== Schema ===")
    print(tool.get_schema_string())
    print("\n=== Test Query ===")
    cols, rows, err = tool.execute_query("SELECT COUNT(*) as cnt FROM Orders")
    print(f"Columns: {cols}")
    print(f"Rows: {rows}")
    print(f"Error: {err}")
    print("\n=== Extract Tables ===")
    test_sql = '''
    SELECT p.ProductName, SUM(od.Quantity) 
    FROM Products p 
    JOIN "Order Details" od ON p.ProductID = od.ProductID
    GROUP BY p.ProductName
    '''
    tables = tool.extract_tables_from_sql(test_sql)
    print(f"Tables found: {tables}")
