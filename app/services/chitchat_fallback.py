# app/services/chitchat_fallback.py

CHITCHAT_RESPONSES = {
    "hi": "Hello! How can I help you today?",
    "hello": "Hi there! What would you like to know?",
    "hey": "Hey! Ready when you are.",
    "how are you": "I'm doing great, ready to answer your data queries!",
    "thank you": "You're welcome!",
    "thanks": "Anytime!",
    "bye": "Goodbye! Talk soon.",
    "who made you": "I was built by Praveen Singh's team!",
}

def is_chitchat_query(query: str) -> bool:
    query = query.lower().strip()
    return any(phrase in query for phrase in CHITCHAT_RESPONSES)

def get_chitchat_response(query: str) -> str:
    query = query.lower().strip()
    for key in CHITCHAT_RESPONSES:
        if key in query:
            return CHITCHAT_RESPONSES[key]
    return "I'm happy to help! Try asking about your data."
