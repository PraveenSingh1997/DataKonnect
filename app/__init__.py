from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_migrate import Migrate
from .models.models import User, db  # Ensure db is defined in models/__init__.py or models/models.py
from app.routes.auth import auth_bp
from app.routes.chat import chat_bp

# Initialize extensions
login_manager = LoginManager()
login_manager.login_view = 'auth.login'  # Redirects to auth.login when login_required fails
migrate = Migrate()

def load_user(user_id):
    """
    Required by Flask-Login to reload the user object from the user ID stored in the session.
    """
    return User.query.get(int(user_id))

login_manager.user_loader(load_user)

def create_app():
    app = Flask(__name__)
    app.config.from_object('config.Config')  # Make sure you have a config.py with class Config

    # Initialize extensions
    db.init_app(app)
    
    migrate.init_app(app, db)
    login_manager.init_app(app)

    # Register Blueprints
    app.register_blueprint(auth_bp, url_prefix='/auth')  # Routes like /auth/login, /auth/register
    app.register_blueprint(chat_bp)                      # Routes like / and /chat

    return app
