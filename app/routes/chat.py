# app/routes/chat.py

from flask import Blueprint, render_template, request, jsonify, current_app
from flask_login import login_required, current_user
from ..models.models import Chat, db
from ..services.llm_service import get_bot_response

chat_bp = Blueprint('chat', __name__)

@chat_bp.route('/', methods=['GET'])
def home():
    """
    Public landing page for the chat assistant.
    """
    return render_template('index.html')

@chat_bp.route('/chat', methods=['GET', 'POST'])
@login_required
def chat():
    """
    Authenticated chat interface. Handles user messages and bot responses.
    """
    if request.method == 'POST':
        user_input = request.form.get('message', '').strip()
        if not user_input:
            return jsonify({ "error": "No message provided" }), 400

        try:
            # 1) Call your LLM service
            response_data = get_bot_response(user_input, current_user.id, user_input)

            # 2) Extract data from response
            llm_reply   = response_data.get("llm_reply", "⚠️ No reply from bot")
            sql_query   = response_data.get("sql_query")
            sql_results = response_data.get("sql_results")
            chart       = response_data.get("chart")

            # 3) Persist chat record
            new_chat = Chat(
                user_id      = current_user.id,
                user_message = user_input,
                bot_reply    = llm_reply,
                
            )
            db.session.add(new_chat)
            db.session.commit()

            # 4) Build payload including optional chart
            payload = {
                "chat_id"    : response_data.get("chat_id"),
                "llm_reply"  : llm_reply,
                "sql_query"  : sql_query,
                "sql_results": sql_results
            }
            if chart is not None:
                payload["chart"] = chart
            print("0000000000000000000000000000000000000000000000000000")
            print(payload)

            return jsonify(payload), 200

        except Exception:
            current_app.logger.exception("Error handling /chat POST")
            # Return 503 so frontend retry logic will kick in
            return (
                jsonify({ "error": "Service temporarily unavailable. Please retry." }),
                503
            )

    # GET → render history
    history = Chat.query.filter_by(user_id=current_user.id).order_by(Chat.id).all()
    return render_template('chat.html', history=history)

@chat_bp.route('/submit-feedback', methods=['POST'])
@login_required
def submit_feedback():
    """
    Record user feedback (👍/👎) on a specific chat bubble.
    """
    data = request.get_json() or {}
    chat_id  = data.get("chat_id")
    feedback = data.get("feedback")

    if feedback not in ('positive', 'negative'):
        return jsonify({ "error": "Invalid feedback value" }), 400

    chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first()
    if not chat:
        return jsonify({ "error": "Chat not found" }), 404

    chat.feedback = feedback
    db.session.commit()
    return jsonify({ "message": "Feedback recorded" }), 200
