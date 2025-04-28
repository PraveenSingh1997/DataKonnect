from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime


db = SQLAlchemy()


class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)

    # Relationship to Chat
    chats = db.relationship('Chat', back_populates='user', lazy='dynamic')

    def set_password(self, password: str) -> None:
        """
        Hash and store the given password.
        """
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        """
        Verify the given password against the stored hash.
        """
        return check_password_hash(self.password_hash, password)
    

class Chat(db.Model):
    __tablename__ = 'chats'
    id           = db.Column(db.Integer, primary_key=True)
    user_id      = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    user         = db.relationship('User', back_populates='chats')
    user_message = db.Column(db.Text, nullable=False)
    bot_reply    = db.Column(db.Text, nullable=False)
    feedback     = db.Column(db.String(10))  # 'positive' or 'negative'
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)
