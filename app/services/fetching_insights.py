import os
import json
import requests
from flask import Blueprint, request, render_template, Response, abort
import logging
from typing import List, Dict, Any
from requests.exceptions import HTTPError

# Configure logging
DEFAULT_LEVEL = os.getenv('LOG_LEVEL', 'DEBUG').upper()
logging.basicConfig(
    level=getattr(logging, DEFAULT_LEVEL, logging.DEBUG),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# LLM configuration
API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE = os.getenv("OPENAI_API_BASE")
DEFAULT_MODEL = os.getenv("INSIGHTS_MODEL", "deepseek-r1")
DEFAULT_TEMPERATURE = float(os.getenv("INSIGHTS_TEMPERATURE", "0.7"))

insights_bp = Blueprint('insights', __name__, template_folder='templates')


def _call_llm(messages: List[Dict[str, Any]], model: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE) -> str:
    """
    Internal helper to send a streaming chat completion request to the LLM.
    """
    if not API_KEY or not API_BASE:
        raise EnvironmentError("OPENAI_API_KEY and OPENAI_API_BASE must be set.")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
        "max_tokens": 2000
    }
    resp = requests.post(f"{API_BASE}/v1/chat/completions", headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def generate_insights_with_llm(sql_results: List[Dict[str, Any]]) -> str:
    """
    Generate a Markdown report from raw SQL results using the LLM.
    """
    if not sql_results:
        return "**No data available to analyze.**"

    # Prompt LLM to output Markdown only
    system_prompt = (
    "You are a data-savvy analyst. Given these raw SQL query results in JSON, "
    "produce a Markdown report with the following structure and nothing else:\n\n"
    "# {{TITLE}}\n"
    "*{{Subtitle (optional)}}*\n\n"
    "## Key Insights\n"
    "- (3–5 bullet points summarizing key takeaways)\n\n"
    "## Summary Statistics\n"
    "| Metric | Value |\n"
    "| ------ | ----- |\n"
    "| ...    | ...   |\n\n"
    "## Charts\n"
    "For each relevant metric or trend, embed a Chart.js configuration block. "
    "Wrap each chart in a `<canvas>` tag and include a `<script>` that:\n"
    "1. Selects the canvas by ID.\n"
    "2. Initializes a new `Chart(ctx, { type: ..., data: {...}, options: {...} })`.\n\n"
    "## Follow-Up Questions\n"
    "1. **Question:** ...  \n"
    "   **Viz:** Reference which Chart.js chart you’d add (e.g. line, bar)  \n"
    "   **Detail:** ...\n\n"
    "Include forward-looking commentary when trends are clear."
)


    user_content = json.dumps(sql_results, default=str)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Here are the query results:\n{user_content}"}
    ]
    return _call_llm(messages)


@insights_bp.route('/insights', methods=['GET'])
def insights_page():
    # Render the HTML container page
    return render_template('insights.html')


@insights_bp.route('/api/insights', methods=['POST'])
def insights_api():
    """
    Endpoint to generate and return insights as Markdown.
    """
    try:
        payload = [
                        {
                            "sales_month": "2024-01-01T00:00:00.000+00:00",
                            "total_net_sales": 27029716545.180187
                        },
                        {
                            "sales_month": "2024-02-01T00:00:00.000+00:00",
                            "total_net_sales": 25932368254.673557
                        },
                        {
                            "sales_month": "2024-03-01T00:00:00.000+00:00",
                            "total_net_sales": 48149092944.60048
                        },
                        {
                            "sales_month": "2024-04-01T00:00:00.000+00:00",
                            "total_net_sales": 44165601826.0013
                        },
                        {
                            "sales_month": "2024-05-01T00:00:00.000+00:00",
                            "total_net_sales": 31620243398.2367
                        },
                        {
                            "sales_month": "2024-06-01T00:00:00.000+00:00",
                            "total_net_sales": 29529011505.0978
                        },
                        {
                            "sales_month": "2024-07-01T00:00:00.000+00:00",
                            "total_net_sales": 71790758626.9312
                        },
                        {
                            "sales_month": "2024-08-01T00:00:00.000+00:00",
                            "total_net_sales": 86965222943.5114
                        },
                        {
                            "sales_month": "2024-09-01T00:00:00.000+00:00",
                            "total_net_sales": 87931624821.1783
                        },
                        {
                            "sales_month": "2024-10-01T00:00:00.000+00:00",
                            "total_net_sales": 82689609729.3379
                        },
                        {
                            "sales_month": "2024-11-01T00:00:00.000+00:00",
                            "total_net_sales": 58295797015.9402
                        },
                        {
                            "sales_month": "2024-12-01T00:00:00.000+00:00",
                            "total_net_sales": 55160466743.3623
                        }
                        ]
        #payload = request.get_json(silent=True) or []
        if not isinstance(payload, list):
            raise ValueError("Expected a JSON array of SQL result objects.")
    except Exception as e:
        logger.error("Invalid request payload: %s", e)
        abort(400, description="Invalid JSON payload; expected a list of objects.")

    try:
        report_md = generate_insights_with_llm(payload)
        logger.debug("Generated Markdown report: %s", report_md)
    except HTTPError as e:
        logger.exception("LLM service error")
        abort(502, description=f"LLM service error: {e}")
    except Exception as e:
        logger.exception("Unexpected error calling LLM")
        abort(500, description=f"Unexpected LLM error: {e}")

    # Return plain Markdown
    return Response(report_md + "\n", status=200, mimetype='text/markdown')
