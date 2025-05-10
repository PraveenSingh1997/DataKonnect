
import os
import requests

API_KEY  = os.getenv("OPENAI_API_KEY", "sk-OFhKUP9G7djOzSIa6SEKKQ")
API_BASE = os.getenv("OPENAI_API_BASE", "https://lmlitellm.landmarkgroup.com")
URL      = f"{API_BASE}/v1/chat/completions"

payload = {
    "model": "deepseek-r1",
    "messages": [
        {"role": "system", "content": "health check"},
        {"role": "user",   "content": "ping"}
    ],
    "temperature": 0.0,
    "max_tokens": 1
}

resp = requests.post(
    URL,
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json"
    },
    json=payload,
    timeout=10
)

if resp.ok:
    data = resp.json()
    print("✅ OK:", data["choices"][0]["message"]["content"].strip())
else:
    print(f"❌ Error {resp.status_code}:", resp.text)
