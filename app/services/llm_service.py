import os
import requests
import re , json
import logging
from langchain.vectorstores.base import VectorStoreRetriever
from app.services.vector_store import get_vectorstore_retriever
from app.services.sql_runner import run_sql_query
from app.models.models import Chat, db
from datetime import datetime
ERROR_LOG_FILE = os.getenv("SQL_ERROR_LOG_FILE", "sql_errors.json")

# Configure logging
DEFAULT_LEVEL = os.getenv('LOG_LEVEL', 'DEBUG').upper()
logging.basicConfig(
    level=getattr(logging, DEFAULT_LEVEL, logging.DEBUG),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set up environment variables
os.environ.setdefault("OPENAI_API_KEY", "sk-OFhKUP9G7djOzSIa6SEKKQ")
os.environ.setdefault("OPENAI_API_BASE", "https://lmlitellm.landmarkgroup.com/")
api_key = os.environ["OPENAI_API_KEY"]
api_base = os.environ["OPENAI_API_BASE"]

# Retry configuration
MAX_SQL_RETRIES = 2 # number of retry attempts on SQL errors
retriever: VectorStoreRetriever = get_vectorstore_retriever()

def init_service():
    """
    Initialize and validate LLM service configuration, environment, and dependencies.
    """
    logger.info("Initializing LLM Service...")
    # Check environment variables
    missing = [var for var in ("OPENAI_API_KEY", "OPENAI_API_BASE") if not os.environ.get(var)]
    if missing:
        logger.error(f"Missing environment variables: {missing}")
        raise EnvironmentError(f"Missing environment variables: {', '.join(missing)}")

    # Test LLM API connectivity
    try:
        logger.debug(f"Testing LLM endpoint at {api_base}/v1/models")
        resp = requests.get(
            f"{api_base}/v1/models",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        resp.raise_for_status()
        logger.info("LLM API connectivity verified")
    except Exception as e:
        logger.error(f"LLM API connectivity failed: {e}")
        raise

    # Initialize and test vector store retriever
    try:
        logger.debug("Initializing vector store retriever for health check")
        
        if not isinstance(retriever, VectorStoreRetriever):
            raise TypeError("Retrieved instance is not a VectorStoreRetriever")
        logger.info("Vector store retriever initialized successfully")
    except Exception as e:
        logger.error(f"Vector store retriever initialization failed: {e}")
        raise

    logger.info("LLM Service initialization complete")
    return True


# Perform initialization check on module load
try:
    init_service()
except Exception as e:
    logger.critical(f"Service failed to initialize: {e}")
    # Continue with degraded functionality if needed

# Initialize retriever after health check
logger.info("Instantiating vector store retriever for runtime")
#retriever: VectorStoreRetriever = get_vectorstore_retriever()
def is_time_column(name, sample_value):
    # crude check: name contains date/time or sample parses as date
    if re.search(r"date|dt_|time", name, re.IGNORECASE):
        return True
    try:
        # try parsing ISO-like strings
        if isinstance(sample_value, str):
            datetime.fromisoformat(sample_value)
            return True
    except Exception:
        pass
    return False
def log_sql_error(question: str, error: str):
    """Append a JSON line with the question and error message."""
    entry = {
        "question": question,
        "error": error
    }
    try:
        with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as log_exc:
        logger.error(f"Failed to write SQL error log: {log_exc}")



def is_chitchat_doc(doc):
    tag = doc.metadata.get("tag", "").lower()
    logger.debug(f"Checking if document with tag '{tag}' is chitchat")
    return tag == "chitchat"


def extract_sql_from_text(text):
    logger.debug("Extracting SQL from LLM response")
    pattern = r"```(?:sql)?\s*(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        sql = match.group(1).strip()
        logger.debug(f"Extracted SQL: {sql}")
        return sql
    cleaned = text.strip()
    logger.debug(f"No code fences found, returning cleaned text: {cleaned}")
    return cleaned


def has_error(results):
    if not isinstance(results, list):
        return False
    error_found = any(isinstance(r, str) and "Error" in r for r in results)
    logger.debug(f"Error in SQL results: {error_found}")
    return error_found


def get_bot_response(message, user_id, user_message):
    logger.info(f"get_bot_response called with message: {message}")
    # Step 1: Retrieve context documents
    try:
        print(message)
        relevant_chunks = retriever.invoke(message)
        print(relevant_chunks)
    except Exception as e:
        logger.warning(f"Retrieval failed, returning empty list: {e}")
        relevant_chunks = []
    logger.debug(f"Relevant chunks retrieved: {relevant_chunks}")

    if not relevant_chunks:
        logger.warning("No relevant chunks found for message")
        return {
            "llm_reply": "🤖 I couldn't find anything related to that. Try asking a data-specific question.",
            "sql_query": None,
            "sql_results": None
        }
    print("step 1 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # Step 2: Handle chitchat
    top_doc = relevant_chunks[0]
    if is_chitchat_doc(top_doc):
        logger.info("Top document is chitchat, using fallback conversation logic")
        chitchat_context = "\n".join(
            doc.page_content for doc in relevant_chunks if is_chitchat_doc(doc)
        )

        messages = [
            {"role": "system", "content": "You are a friendly assistant that replies to greetings and casual conversations."},
            {"role": "system", "content": f"Chitchat Context:\n{chitchat_context}"},
            {"role": "user",   "content": message}
        ]
        data = {"model": "depseek-r1", "messages": messages, "temperature": 0.3, "max_tokens": 300}

        try:
            resp = requests.post(
                f"{api_base}/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=data
            )
            resp.raise_for_status()
            llm_reply = resp.json()["choices"][0]["message"]["content"].strip()

            new_chat = Chat(user_id=user_id, user_message=user_message, bot_reply=llm_reply)
            db.session.add(new_chat)
            db.session.commit()
            return {"chat_id": new_chat.id, "llm_reply": llm_reply, "sql_query": None, "sql_results": None}
        except Exception as e:
            logger.error(f"Error during chitchat fallback: {e}")
            return {"llm_reply": f"❌ Error (chitchat): {e}", "sql_query": None, "sql_results": None}

    # Step 3: SQL-oriented LLM
    context = "\n".join(doc.page_content for doc in relevant_chunks)
    '''base_messages = [
        {"role": "system", "content": (
            "You must generate databricks SQL using only the tables like fct_sls_cnsld and columns provided in the context below. "
            "DT_KEY is a bigint and should be joined via DIM_DT for date filters. Do not guess columns not in context."
            "The lmdata_lh_gld_pprd.er_rtl_dm.FCT_SLS_CNSLD table includes a wide range of columns supporting sales analytics. It contains key fields like DLY_SLS_KEY, DT_KEY is a big int in the format YYYYMMDD , , ITM_KEY, LOC_KEY, and ITM_SSN_KEY, which serve as surrogate or foreign keys for joining with respective dimension tables. Transaction attributes such as TX_TYP_CD, CNCPT_KEY, and CRRNCY_CD define the type, concept, and currency of the transaction. Sales volume is captured through RTL_QTY, HH_RTL_QTY, and FF_RTL_QTY, along with prior year comparisons like RTL_QTY_TRDNG_LY and RTL_QTY_FSCL_LY. Financials are detailed with fields like GRS_SLS_AMT, NET_SLS_AMT, DSCNT_AMT, CPN_AMT, and their household (HH_) and fast fashion (FF_) counterparts. Unit pricing data appears in columns such as UNT_RTL, ORGNL_UNT_RTL, REG_UNT_RTL, and REG_UNT_RTL_VAT_INC. Profitability metrics include COGS_1, COGS_2, GRS_MRGN_1, GRS_MRGN_2, and INTK_MRGN, while margin fields such as MRKUP, MRKDN, and AVG_SLS_PR_SQ_FT offer insights into pricing strategy and efficiency. Tax information is stored in TAX_1_AMT, TAX_2_AMT, TAX_3_AMT, TAX_EXMPT, and price-related indicators like PRC_FLG and MRK_FLG. Operational metadata includes SEC_GRP_CD, INTGRTN_ID, SRC_SYS_CD, SRC_SYS_VER, BTCH_ID, CRT_DTTM, and LST_MODFD_DTTM. Additional fields such as TRDNG_YR_AGO_DT_KEY, FSCL_YR_AGO_DT_KEY, LOC_CD, ITM_CD, FST_SLD_PRC, FST_SLD_DT, and SLS_PGV_DISTRB_VAL support historical, store, and item-level tracking. Altogether, this table offers rich granularity for measuring sales, trends, profitability, and store-item interactions over time.lmdata_lh_gld_pprd.er_rtl_dm.FCT_SLS_CNSLD joins with lmdata_lh_gld_pprd.er_rtl_dm.DIM_DT using DT_KEY is a big int in the format YYYYMMDD ,  to support time-based aggregations and YOY comparisons. It links with DIM_ITM, DIM_ITM_SSN, and DIM_LOC using ITM_KEY, ITM_SSN_KEY, and LOC_KEY respectively to enable granular analysis at the item-location-date level. Additional relationships with DIM_CNCPT, lmdata_lh_gld_pprd.er_rtl_dm.DIM_STR_CNPT_AREA_MGR, DIM_STATIC_EXCH_CURRENCY_AED, lmdata_lh_gld_pprd.er_rtl_dm.DIM_ITM_LOC, and DIM_STR_CURR provide contextual enrichment around store hierarchy, pricing, currency normalization, and sales performance. This table serves as the core for daily sales reporting and supports financial, operational, and strategic KPIs across concepts, channels, and territories."
        )},
        {"role": "system", "content": f"Context:\n{context}"},
        {"role": "user",   "content": message}
    ]

    try:
        llm_reply = None
        sql_query = None
        sql_results = None
        chart = None

        for attempt in range(MAX_SQL_RETRIES + 1):
            resp = requests.post(
                f"{api_base}/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": "landmark-gpt-4o-mini", "messages": base_messages, "temperature": 0.7, "max_tokens": 1000}
            )
            resp.raise_for_status()
            llm_reply = resp.json()["choices"][0]["message"]["content"].strip()

            # Extract and run SQL
            sql_query = extract_sql_from_text(llm_reply)
            if sql_query:'''
    
    error_message = None
    print("step 2 +++++++++++++++++++++++++++++++++++++++++++++++++++")

# build the static part of your messages once
    base_messages = [
        {"role": "system", "content": (
            "You must generate SQL using only the tables and columns provided in the context below. "
            "DT_KEY is a bigint and should be joined via DIM_DT for date filters. Do not guess columns not in context."
            "The lmdata_lh_gld_pprd.er_rtl_dm.FCT_SLS_CNSLD table includes a wide range of columns supporting sales analytics. It contains key fields like DLY_SLS_KEY, DT_KEY is a big int in the format YYYYMMDD , , ITM_KEY, LOC_KEY, and ITM_SSN_KEY, which serve as surrogate or foreign keys for joining with respective dimension tables. Transaction attributes such as TX_TYP_CD, CNCPT_KEY, and CRRNCY_CD define the type, concept, and currency of the transaction. Sales volume is captured through RTL_QTY, HH_RTL_QTY, and FF_RTL_QTY, along with prior year comparisons like RTL_QTY_TRDNG_LY and RTL_QTY_FSCL_LY. Financials are detailed with fields like GRS_SLS_AMT, NET_SLS_AMT, DSCNT_AMT, CPN_AMT, and their household (HH_) and fast fashion (FF_) counterparts. Unit pricing data appears in columns such as UNT_RTL, ORGNL_UNT_RTL, REG_UNT_RTL, and REG_UNT_RTL_VAT_INC. Profitability metrics include COGS_1, COGS_2, GRS_MRGN_1, GRS_MRGN_2, and INTK_MRGN, while margin fields such as MRKUP, MRKDN, and AVG_SLS_PR_SQ_FT offer insights into pricing strategy and efficiency. Tax information is stored in TAX_1_AMT, TAX_2_AMT, TAX_3_AMT, TAX_EXMPT, and price-related indicators like PRC_FLG and MRK_FLG. Operational metadata includes SEC_GRP_CD, INTGRTN_ID, SRC_SYS_CD, SRC_SYS_VER, BTCH_ID, CRT_DTTM, and LST_MODFD_DTTM. Additional fields such as TRDNG_YR_AGO_DT_KEY, FSCL_YR_AGO_DT_KEY, LOC_CD, ITM_CD, FST_SLD_PRC, FST_SLD_DT, and SLS_PGV_DISTRB_VAL support historical, store, and item-level tracking. Altogether, this table offers rich granularity for measuring sales, trends, profitability, and store-item interactions over time.lmdata_lh_gld_pprd.er_rtl_dm.FCT_SLS_CNSLD joins with lmdata_lh_gld_pprd.er_rtl_dm.DIM_DT using DT_KEY is a big int in the format YYYYMMDD ,  to support time-based aggregations and YOY comparisons. It links with DIM_ITM, DIM_ITM_SSN, and DIM_LOC using ITM_KEY, ITM_SSN_KEY, and LOC_KEY respectively to enable granular analysis at the item-location-date level. Additional relationships with DIM_CNCPT, lmdata_lh_gld_pprd.er_rtl_dm.DIM_STR_CNPT_AREA_MGR, DIM_STATIC_EXCH_CURRENCY_AED, lmdata_lh_gld_pprd.er_rtl_dm.DIM_ITM_LOC, and DIM_STR_CURR provide contextual enrichment around store hierarchy, pricing, currency normalization, and sales performance. This table serves as the core for daily sales reporting and supports financial, operational, and strategic KPIs across concepts, channels, and territories."
            "decimal should be till 2 digits ."
        )},
        {"role": "system", "content": f"Context:\n{context}"},
        {"role": "user",   "content": message}
    ]

    llm_reply = None
    sql_query = None
    sql_results = None

    for attempt in range(MAX_SQL_RETRIES + 1):
        # rebuild messages for this attempt
        messages = list(base_messages)
        if error_message:
            # insert right before the user’s question so the LLM sees the failure
            messages.insert(
                -1,
                {"role": "system",
                "content": f"Note: The previous SQL execution failed with error:\n{error_message}"}
            )           

        # call the LLM
        print("step 3 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ")
        print(messages)
        resp = requests.post(
            f"{api_base}v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "deepseek-r1",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1000
            }
        )
        print(resp)
        resp.raise_for_status()
        llm_reply = resp.json()["choices"][0]["message"]["content"].strip()
        print(llm_reply)

        # pull out the SQL
        sql_query = extract_sql_from_text(llm_reply)
        if not sql_query:
            break  # nothing to run

        # try to run it
        try:
            sql_results = run_sql_query(sql_query)
            # if your runner returns a list with an "Error" string:
            if isinstance(sql_results, list) and sql_results and "Error" in sql_results[0]:
                raise RuntimeError(sql_results[0])

            # success!
            break

        except Exception as e:
            error_message = str(e)
            logger.warning(f"SQL attempt {attempt+1} failed: {error_message}")
            # if this was our last allowed retry, bail out of the loop
            log_sql_error(message, error_message)
            if attempt >= MAX_SQL_RETRIES:
                logger.error("Max SQL retries reached, aborting further retries.")
                break
            # otherwise, loop around and try again
            continue

# at this point you have llm_reply, sql_query, and either sql_results or final error_message
            
                
               

            
                
    '''chart = None
    print(isinstance(sql_results, list) and len(sql_results))
    if isinstance(sql_results, list) and len(sql_results) > 0:
        cols = list(sql_results[0].keys())
        if len(cols) == 2:
            x_key, y_key = cols
            labels = [row[x_key] for row in sql_results]
            values = [row[y_key] for row in sql_results]
            chart_type = "line" if re.search(r"date|dt_|time", x_key, re.IGNORECASE) else "bar"
            print(chart_type)
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            chart = {
                "type": chart_type,
                "data": {
                    "labels": labels,
                    "datasets": [{
                        "label": y_key,
                        "data": values, 
                        "borderWidth": 1
                    }]
                },
                "options": {"responsive": True, "scales": {"y": {"beginAtZero": True}}}
            }

        # Retry feedback
        base_messages.append({"role": "system", "content": f"The SQL returned errors:\n{sql_results}\nPlease fix."})

    # Save chat and return
    new_chat = Chat(user_id=user_id, user_message=user_message, bot_reply=llm_reply)
    db.session.add(new_chat)
    db.session.commit()

    payload = {"chat_id": new_chat.id, "llm_reply": llm_reply, "sql_query": sql_query, "sql_results": sql_results}
    if chart is not None:
        payload["chart"] = chart
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print(payload)'''
    
    chart = None
    print ( isinstance(sql_results, list) and len(sql_results) > 0 and not has_error(sql_results))
# only attempt to build a chart if we got back a list of rows and no SQL errors
    '''if isinstance(sql_results, list) and len(sql_results) > 0 and not has_error(sql_results):
        cols = list(sql_results[0].keys())
        if len(cols) == 2:
            x_key, y_key = cols
            labels = [row[x_key] for row in sql_results]
            values = [row[y_key] for row in sql_results]
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            chart_type = "line" if re.search(r"date|dt_|time", x_key, re.IGNORECASE) else "bar"
            chart = {
                "type": chart_type,
                "data": {
                    "labels": labels,
                    "datasets": [{
                        "label": y_key,
                        "data": values,
                        "borderWidth": 1
                    }]
                },
                "options": {"responsive": True, "scales": {"y": {"beginAtZero": True}}}
            }
    else:
        # we either got an error string back or no rows—skip chart
        logger.debug(f"Skipping chart generation, sql_results={sql_results!r}")'''
    
    
    



    if isinstance(sql_results, list) and len(sql_results) > 0 and not has_error(sql_results):
    # get column names
        cols = list(sql_results[0].keys())

        # decide x_key vs y_keys
        first_col, first_val = cols[0], sql_results[0][cols[0]]
        if is_time_column(first_col, first_val):
            x_key = first_col
            y_keys = cols[1:]
        elif all(isinstance(sql_results[0][c], (int, float)) for c in cols):
            # no explicit x, use index
            x_key = None
            y_keys = cols
        else:
            # treat first col as category axis
            x_key = first_col
            y_keys = cols[1:]

        # build labels
        if x_key:
            labels = [row[x_key] for row in sql_results]
        else:
            labels = list(range(1, len(sql_results) + 1))

        # build one dataset per y_key
        datasets = []
        for y in y_keys:
            datasets.append({
                "label": y,
                "data": [row[y] for row in sql_results],
                "borderWidth": 1,
                # you can add styling here per-series if desired
            })

        # choose overall chart type
        chart_type = "line" if x_key and is_time_column(x_key, first_val) else "bar"

        chart = {
            "type": chart_type,
            "data": {
                "labels": labels,
                "datasets": datasets
            },
            "options": {
                "responsive": True,
                "scales": {
                    "y": {"beginAtZero": True}
                }
            }
        }
        print("Generated chart config:", chart)
    else:
        # skip if error or no data
        logger.debug(f"Skipping chart generation, sql_results={sql_results!r}")


    # Save chat and return
    new_chat = Chat(user_id=user_id, user_message=user_message, bot_reply=llm_reply)
    db.session.add(new_chat)
    db.session.commit()

    payload = {
        "chat_id": new_chat.id,
        "llm_reply": llm_reply,
        "sql_query": sql_query,
        "sql_results": sql_results
    }
    if chart is not None:
        payload["chart"] = chart

    return payload
    

