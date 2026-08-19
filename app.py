from flask import Flask, render_template, request, jsonify,session, flash,redirect,url_for
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch
from werkzeug.security import check_password_hash, generate_password_hash
from helper import apology, login_required
import sqlite3
from flask_session import Session
from datetime import datetime 
from pymongo import MongoClient
from bson import ObjectId

app = Flask(__name__)

app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)


def get_database():
    conn = sqlite3.connect('./auth.db')
    conn.row_factory = sqlite3.Row
    return conn

client = MongoClient("mongodb://localhost:27017/")
db = client["Kaarshika"] 
chats_collection = db["chats"]
farm_states_collection = db["farm_context"]

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

@app.route('/chat/<chat_id>')
@login_required
def chat(chat_id):

    user_id = session.get("user_id")

    # Validate MongoDB ObjectId
    try:
        object_id = ObjectId(chat_id)

    except Exception:
        return redirect(url_for('new_chat_page'))

    # Find chat belonging to logged-in user
    chat_doc = chats_collection.find_one({
        '_id': object_id,
        'user_id': user_id
    })

    if not chat_doc:
        return "Chat not found", 404

    farm_state = farm_states_collection.find_one({
        'user_id': user_id
    })

    messages = chat_doc.get('messages', [])

    return render_template(
        'chat.html',
        messages=messages,
        is_new_chat=(len(messages) == 0),
        chat_id=chat_id,
        farm_state=farm_state
    )

@app.route('/new-chat', methods=['GET'])
@login_required
def new_chat_page():

    chat_id = ObjectId()

    new_chat = {
        '_id': chat_id,
        'user_id': session.get("user_id"),
        'title': 'New Chat',
        'messages': [],
        'created_at': datetime.utcnow(),
        'updated_at': datetime.utcnow()
    }

    chats_collection.insert_one(new_chat)

    return redirect(
        url_for(
            'chat',
            chat_id=str(chat_id)
        )
    )
@app.route('/farm_state', methods=['GET', 'POST'])
@login_required
def farm_state():

    user_id = session.get("user_id") 
    if request.method == 'GET':

        farm_data = farm_states_collection.find_one({
            'user_id': user_id
        })

        return render_template(
            'state.html',
            farm_state=farm_data
        )

    location = request.form.get('location', '').strip()
    crop = request.form.get('crop', '').strip()
    stage = request.form.get('stage', '').strip()
    acres = request.form.get('acres', '').strip()
    soil_moisture = request.form.get('soil_moisture', '').strip()

    if not location or not crop or not stage or not acres:
        return "All fields are required.", 400

    try:
        acres = float(acres)

        if acres <= 0:
            return "Acres must be greater than 0.", 400

    except ValueError:
        return "Invalid land area.", 400

    farm_states_collection.update_one(

        {
            'user_id': user_id
        },

        {
            '$set': {
                'location': location,
                'crop': crop,
                'stage': stage,
                'acres': acres,
                'soil_moisture': soil_moisture,
                'updated_at': datetime.utcnow()
            },

            '$setOnInsert': {
                'user_id': user_id,
                'created_at': datetime.utcnow()
            }
        },

        upsert=True
    )

    return redirect(url_for('new_chat_page'))

@app.route('/get_response', methods=['POST'])
@login_required
def get_response():

    data = request.get_json()

    user_message = data.get('message', '').strip()
    chat_id = data.get('chat_id')

    if not user_message:
        return jsonify({
            'reply': 'Please type something.'
        }), 400

    if not chat_id:
        return jsonify({
            'error': 'Chat ID is required.'

        }), 400

    try:
        object_id = ObjectId(chat_id)

    except Exception:
        return jsonify({
            'error': 'Invalid chat ID.'
        }), 400


    chat_doc = chats_collection.find_one({
        '_id': object_id,
        'user_id': session.get("user_id")
    })

    if not chat_doc:
        return jsonify({
            'error': 'Chat not found.'
        }), 404


    inputs = tokenizer(
        user_message,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {
        k: v.to(DEVICE)
        for k, v in inputs.items()
    }


    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(
            outputs.logits,
            dim=-1
        )
        predicted_id = torch.argmax(
            probs,
            dim=-1
        ).item()


    intent = LABEL_NAMES[predicted_id]


    now = datetime.utcnow()

    user_msg_doc = {
        'sender': 'user',
        'text': user_message,
        'timestamp': now
    }

    bot_msg_doc = {
        'sender': 'bot',
        'text': intent,
        'timestamp': datetime.utcnow()
    }


    messages = chat_doc.get('messages', [])
    is_first_message = len(messages) == 0


    if is_first_message:
        title = ' '.join(
            user_message.split()
        )
        # Limit title to 50 characters
        if len(title) > 50:
            title = title[:50].rstrip() + '...'
    else:
        title = chat_doc.get(
            'title',
            'New Chat'
        )

    chats_collection.update_one(
        {
            '_id': object_id,
            'user_id': session.get("user_id")
        },

        {
            '$set': {
                'title': title,
                'updated_at': datetime.utcnow()
            },

            '$push': {
                'messages': {
                    '$each': [
                        user_msg_doc,
                        bot_msg_doc
                    ]
                }
            }
        }
    )

    return jsonify({
        'reply': intent,
        'chat_id': chat_id,
        'title': title
    })
  
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
