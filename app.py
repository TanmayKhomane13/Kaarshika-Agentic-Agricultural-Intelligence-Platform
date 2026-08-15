from flask import Flask, render_template, jsonify # session, request, redirect, url_for, flash
#from flask_session import Session
#from datetime import datetime
#from werkzeug.security import check_password_hash, generate_password_hash
from helper import apology, login_required

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")



@app.route('/get_response', methods=['POST'])
def get_response():
    user_message = request.json.get('message', '')
    
    # Placeholder for logic:
    bot_reply = f"Echo: {user_message}" 
    
    return jsonify({'reply': bot_reply})