from flask import Flask, render_template, request, jsonify,session, flash,redirect,url_for
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch
from werkzeug.security import check_password_hash, generate_password_hash
from helper import apology, login_required
import sqlite3
from flask_session import Session
import datetime, timezone
from pymongo import MongoClient
import json
from bson.objectid import ObjectId

app = Flask(__name__)

app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)


def get_database():
    conn = sqlite3.connect('./auth.db')
    conn.row_factory = sqlite3.Row
    return conn


# ====================== MODEL LOADING ======================
BASE_MODEL = "distilbert-base-uncased"
ADAPTER_PATH = "./AI/Notebooks/classifier_lora"

NUM_LABELS = 2
LABEL_NAMES = [
    "DECISION",
    "INFORMATION"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading base model + LoRA adapter...")
base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=NUM_LABELS
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.to(DEVICE)
model.eval()
print("Model loaded successfully!")
# ===========================================================


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    """Log user in"""
    session.clear()
    if request.method == "POST":
        username = request.form.get("name")
        password = request.form.get("password")

        if not username or not password:
            flash("Username and password are required", "danger")
            return redirect(url_for("login"))

        db = get_database()
        try:
            user = db.execute(
                "SELECT id, name, password FROM users WHERE name = ? ",
                (username,)
            ).fetchone()

            if not user:
                flash("User not found", "danger")
                return redirect(url_for("login"))

            if not check_password_hash(user["password"], password):
                flash("Incorrect password", "danger")
                return redirect(url_for("login"))

            #  Login successful
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]

            flash(f"Welcome back, {user['name']}!", "success")
            
            # Force redirect to home
            return redirect("/")

        except Exception as e:
            flash(f"Error: {str(e)}", "danger")
            return redirect(url_for("login"))
        
        finally:
            db.close()

    return render_template("login.html")

@app.route("/logout")
def logout():
    """Log user out"""

    # Forget any user_id
    session.clear()

    # Redirect user to login form
    return redirect("/")

@app.route("/register", methods=["GET", "POST"])
def register():
    """Register user"""
    if request.method == "POST":
        name = request.form.get("name")
        password = request.form.get("password")
        confirmation = request.form.get("confirmation")

        # Validation
        if not name:
            flash("Name must be provided", "danger")
            return redirect(url_for("register"))

        if not password:
            flash("Password must be provided", "danger")
            return redirect(url_for("register"))

        if password != confirmation:
            flash("Passwords do not match", "danger")
            return redirect(url_for("register"))

        db = get_database()

        try:
            # Check if name or uniquekey already exists
            existing = db.execute(
                "SELECT id FROM users WHERE name = ?", 
                (name,)
            ).fetchone()

            if existing:
                flash("Name is already taken", "danger")
                return redirect(url_for("register"))

            # Hash password
            password_hash = generate_password_hash(password)

            # Insert new user
            db.execute("""
                INSERT INTO users (name, password)
                VALUES (?, ?)
            """, (name, password_hash))

            db.commit()

            # Get the newly created user
            user = db.execute(
                "SELECT id, name FROM users WHERE name = ?", 
                (name)
            ).fetchone()

            # Log user in automatically
            session.clear()
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]

            flash("Registration successful! Welcome to Kaarshika.", "success")
            return redirect(url_for("index"))

        except Exception as e:
            db.rollback()
            flash("An error occurred during registration. Please try again.", "danger")
            return redirect(url_for("register"))

        finally:
            db.close()

    # GET request
    return render_template("register.html")

@app.route('/chat')
@app.route('/chat/<chat_id>')
@login_required
def chat(chat_id=None):
    if chat_id:
        # Opening a PREVIOUS chat -> load its messages, hide the input box
        try:
            chat_doc = chats_collection.find_one({
                '_id': ObjectId(chat_id),
                'user_id': current_user.id
            })
        except Exception:
            chat_doc = None
 
        if not chat_doc:
            # Invalid/foreign chat_id -> fall back to a fresh chat
            return redirect(url_for('chat'))
 
        return render_template(
            "chat.html",
            messages=chat_doc.get('messages', []),
            is_new_chat=False,
            chat_id=chat_id
        )
 
    # No chat_id -> NEW chat, show empty conversation + input box
    return render_template(
        "chat.html",
        messages=None,
        is_new_chat=True,
        chat_id=None
    )
 
 
@app.route('/get_response', methods=['POST'])
@login_required
def get_response():
    data = request.get_json()
    user_message = data.get('message', '').strip()
    chat_id = data.get('chat_id')
 
    if not user_message:
        return jsonify({'reply': "Please type something."})
 
    inputs = tokenizer(
        user_message,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
 
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        predicted_id = torch.argmax(probs, dim=-1).item()
 
    intent = LABEL_NAMES[predicted_id]
 
    user_msg_doc = {
        'sender': 'user',
        'text': user_message,
        'timestamp': datetime.datetime.utcnow()
    }
    bot_msg_doc = {
        'sender': 'bot',
        'text': intent,
        'timestamp': datetime.datetime.utcnow()
    }
 
    if chat_id:
        # Append to an existing chat
        chats_collection.update_one(
            {'_id': ObjectId(chat_id), 'user_id': current_user.id},
            {'$push': {'messages': {'$each': [user_msg_doc, bot_msg_doc]}}}
        )
    else:
        # First message of a brand new chat -> create the document
        new_chat = {
            'user_id': current_user.id,
            'created_at': datetime.datetime.utcnow(),
            'messages': [user_msg_doc, bot_msg_doc]
        }
        result = chats_collection.insert_one(new_chat)
        chat_id = str(result.inserted_id)
 
    return jsonify({
        'reply': intent,       # Only the class name will be shown
        'chat_id': chat_id     # Sent back so the frontend keeps attaching to this chat
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)