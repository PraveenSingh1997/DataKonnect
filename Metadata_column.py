from databricks import sql
from datetime import datetime
import time 
def fetch_columns_for_table(table_name):
    try:
        print(f"🔌 Connecting to Databricks SQL for table `{table_name}`...")
        conn = sql.connect(
            server_hostname="adb-2751553494660337.17.azuredatabricks.net",
            http_path="sql/protocolv1/o/2751553494660337/0218-102529-8sbuupud",
            access_token="dapide57b11d7e94f40d839b384d8f26df8d-2"
        )

        cursor = conn.cursor()
        cursor.execute(f"DESCRIBE TABLE {table_name}")
        results = cursor.fetchall()

        column_names = [row[0] for row in results if row[0] not in ('# col_name', '', None)]
        print(f"✅ Columns in `{table_name}`: {column_names}")
        return column_names

    except Exception as e:
        print(f"❌ Error fetching columns: {e}")
        return []

    finally:
        if 'conn' in locals():
            conn.close()
            print("🔒 Connection closed.")

# Example usage
table_list = [
    "lmdata_lh_gld_prd.bkp_er_rtl_dm.FCT_SLS_CNSLD",
    "lmdata_lh_gld_prd.bkp_er_rtl_dm.DIM_ITM",
    "lmdata_lh_gld_prd.bkp_er_rtl_dm.DIM_LOC",
    "lmdata_lh_gld_prd.bkp_er_rtl_dm.FCT_SLS_EVNT",
    "lmdata_lh_gld_prd.bkp_er_rtl_dm.DIM_STATIC_EXCH_CURRENCY_AED",
    "lmdata_lh_gld_prd.bkp_er_rtl_dm.DIM_ITM_LOC",
    "lmdata_lh_gld_prd.bkp_er_rtl_dm.DIM_STR_CNPT_AREA_MGR",
    "lmdata_lh_gld_prd.bkp_er_rtl_dm.DIM_DT",
    "lmdata_lh_gld_prd.bkp_er_rtl_dm.DIM_CNCPT",
    "lmdata_lh_gld_prd.bkp_er_rtl_dm.DIM_STR_CURR",
    "lmdata_lh_gld_prd.bkp_er_rtl_dm.DIM_ITM_SPLR",
    "lmdata_lh_gld_prd.bkp_er_rtl_dm.DIM_STR_LIKE_NON_LIKE",
    "lmdata_lh_gld_prd.bkp_er_rtl_dm.DIM_ITM_TRRTRY",
    "lmdata_lh_gld_prd.bkp_er_rtl_dm.DIM_FINCE_EVNT_MSTR",
    "lmdata_lh_gld_prd.bkp_er_rtl_dm.DIM_ITM_SSN",

]

rag_metadata = {}

for table in table_list:
     # e.g., FCT_SLS_CNSLD
    columns = fetch_columns_for_table(table)
    rag_metadata[table] = columns
    time.sleep(10)

# Optional: Save as JSON
import json
with open("rag_metadata.json", "w") as f:
    json.dump(rag_metadata, f, indent=4)

print("\n📁 RAG_METADATA:")
print(json.dumps(rag_metadata, indent=4))
