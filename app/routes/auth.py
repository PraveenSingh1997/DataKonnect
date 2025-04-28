from flask import Blueprint, render_template, redirect, url_for, request
from flask_login import login_user, logout_user
from ..models.models import db, User
from ..models.forms import LoginForm, RegisterForm

# Blueprint setup
auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """
    Handle user registration.
    """
    form = RegisterForm()
    if form.validate_on_submit():
        existing_user = User.query.filter_by(username=form.username.data).first()
        if existing_user:
            form.username.errors.append("Username already exists.")
        else:
            user = User(username=form.username.data)
            user.set_password(form.password.data)
            db.session.add(user)
            db.session.commit()
            return redirect(url_for('auth.login'))
    return render_template('register.html', form=form)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """
    Handle user login.
    """
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.check_password(form.password.data):
            login_user(user)
            return redirect(url_for('chat.chat'))
        else:
            form.password.errors.append("Invalid username or password.")
    return render_template('login.html', form=form)

@auth_bp.route('/logout')
def logout():
    """
    Handle user logout.
    """
    logout_user()
    return redirect(url_for('auth.login'))

@auth_bp.route('/index')
def index():
    """
    Optional landing page after login.
    """
    return render_template('index.html')
