from databricks import sql
import os

# It's better to fetch credentials once and reuse them
DATABRICKS_CONFIG = {
    "server_hostname": os.getenv("DATABRICKS_HOST", "adb-2751553494660337.17.azuredatabricks.net"),
    "http_path": os.getenv("DATABRICKS_HTTP_PATH", "sql/protocolv1/o/2751553494660337/0218-102529-8sbuupud"),
    "access_token": os.getenv("DATABRICKS_TOKEN", "dapide57b11d7e94f40d839b384d8f26df8d-2")
}

def run_sql_query(query: str):
    try:
        print(query)
        print("🔌 Connecting to Databricks SQL Warehouse...")
        with sql.connect(**DATABRICKS_CONFIG) as conn:
            print("✅ Connection established.")

            with conn.cursor() as cursor:
                print(f"📤 Executing query:\n{query}")
                cursor.execute(query)

                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                results = [dict(zip(columns, row)) for row in rows]

                print(f"✅ Rows fetched: {len(results)}")
                return results

    except Exception as e:
        print(f"❌ SQL Error occurred: {e}")
        return [f"SQL Error: {e}"]
'''query = "Select * from lmdata_lh_gld_prd.bkp_er_rtl_dm.FCT_SLS_CNSLD limit 1 ; "
run_sql_query(query)'''



'''from databricks import sql
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Config
DATABRICKS_CONFIG = {
    "server_hostname": os.getenv("DATABRICKS_HOST", "adb-2751553494660337.17.azuredatabricks.net"),
    "http_path": os.getenv("DATABRICKS_HTTP_PATH", "sql/protocolv1/o/2751553494660337/0218-102529-8sbuupud"),
    "access_token": os.getenv("DATABRICKS_TOKEN", "dapide57b11d7e94f40d839b384d8f26df8d-2")
}

def _execute_query(query: str):
    try:
        with sql.connect(**DATABRICKS_CONFIG) as conn:
            with conn.cursor() as cursor:
                print(f"📤 Executing query:\n{query}")
                cursor.execute(query)
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                results = [dict(zip(columns, row)) for row in rows]
                print(f"✅ Query finished: {len(results)} rows")
                return {"query": query, "results": results}
    except Exception as e:
        print(f"❌ Error in query: {query}\n{e}")
        return {"query": query, "error": str(e)}

def run_sql_query_concurrently(queries, max_workers=5):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_query = {executor.submit(_execute_query, q): q for q in queries}

        for future in as_completed(future_to_query):
            result = future.result()
            results.append(result)
    
    return results
'''