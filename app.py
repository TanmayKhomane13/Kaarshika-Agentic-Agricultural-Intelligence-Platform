from flask import Flask, render_template, request, jsonify,session, flash,redirect,url_for,Response, stream_with_context
from transformers import AutoTokenizer,AutoModelForCausalLM, AutoModelForSequenceClassification,TextIteratorStreamer
from peft import PeftModel
import torch
from threading import Thread
from werkzeug.security import check_password_hash, generate_password_hash
from helper import apology, login_required
import sqlite3
from flask_session import Session
from datetime import datetime 
from pymongo import MongoClient
from bson import ObjectId
import textwrap
import os
from dotenv import load_dotenv
import requests

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

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

# ====================== MODEL 1: INTENT CLASSIFIER ======================

CLASSIFIER_BASE_MODEL = "distilbert-base-uncased"
CLASSIFIER_ADAPTER_PATH = "./AI/Notebooks/classifier_lora"

NUM_LABELS = 2

LABEL_NAMES = [
    "DECISION",
    "INFORMATION"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading Model 1 tokenizer...")

tokenizer_1 = AutoTokenizer.from_pretrained(
    CLASSIFIER_BASE_MODEL
)

print("Loading Model 1 base model + LoRA adapter...")

base_model_1 = AutoModelForSequenceClassification.from_pretrained(
    CLASSIFIER_BASE_MODEL,
    num_labels=NUM_LABELS
)

model_1 = PeftModel.from_pretrained(
    base_model_1,
    CLASSIFIER_ADAPTER_PATH
)

model_1.to(DEVICE)
model_1.eval()

print("Model 1 loaded successfully!")


# ====================== MODEL 2: RESPONSE GENERATOR ======================

GENERATOR_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
GENERATOR_LORA_PATH = "./model/qwen2.5-3b-lora"

OFFLOAD_DIR = "./content/offload"
os.makedirs(OFFLOAD_DIR, exist_ok=True)

print("Loading Model 2 tokenizer...")

tokenizer_2 = AutoTokenizer.from_pretrained(
    GENERATOR_LORA_PATH
)

print("Loading Model 2 base model...")

base_model_2 = AutoModelForCausalLM.from_pretrained(
    GENERATOR_BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder=OFFLOAD_DIR
)

print("Loading Model 2 LoRA adapter...")

model_2 = PeftModel.from_pretrained(
    base_model_2,
    GENERATOR_LORA_PATH
)

model_2.eval()

print("Model 2 loaded successfully!")

#=======================================================

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

def classify_rainfall(total_mm):
    """
    Classifies total rainfall for today.

    low      : < 10 mm
    moderate : 10 - 50 mm
    high     : > 50 mm
    """

    if total_mm < 10:
        return "low"
    elif total_mm <= 50:
        return "moderate"
    else:
        return "high"

def get_today_rainfall(city):

    url = "https://api.openweathermap.org/data/2.5/forecast"

    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }

    response = requests.get(url, params=params, timeout=10)

    if response.status_code != 200:
        raise Exception(
            f"OpenWeather API error: {response.status_code} "
            f"{response.text}"
        )

    data = response.json()

    # print("City:", data["city"]["name"])
    # print("Forecast entries:", len(data["list"]))

    # Today's date
    today = datetime.now().date()

    total_rainfall = 0.0

    for forecast in data["list"]:

        # Forecast datetime
        forecast_time = datetime.fromtimestamp(
            forecast["dt"]
        )

        # Only consider today's forecast
        if forecast_time.date() != today:
            continue

        # OpenWeather gives rainfall in mm for the
        # preceding 3-hour period.
        rain_data = forecast.get("rain", {})

        rainfall_3h = rain_data.get("3h", 0)

        total_rainfall += rainfall_3h

    rainfall_level = classify_rainfall(total_rainfall)

    return rainfall_level

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

    user_id = session.get("user_id")

    # ==========================================
    # FARM CONTEXT
    # ==========================================

    farm_context_doc = farm_states_collection.find_one({
        'user_id': user_id
    })

    if not farm_context_doc:
        return jsonify({
            'error': 'Farm context not found.'
        }), 404

    farm_context_doc.pop('_id', None)

    location = farm_context_doc.get("location", "Unknown")
    rainfall_data = get_today_rainfall(location)

    farm_context_text = f"""
Crop: {farm_context_doc.get('crop', 'Unknown')}
Stage: {farm_context_doc.get('stage', 'Unknown')}
Soil moisture: {farm_context_doc.get('soil_moisture', 'Unknown')}
Rain probability: {rainfall_data}
Temperature: {farm_context_doc.get('temperature', 'Unknown')}
Humidity: {farm_context_doc.get('humidity', 'Unknown')}
"""

    # ==========================================
    # MODEL 1
    # ==========================================

    inputs = tokenizer_1(
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

        outputs = model_1(**inputs)

        predicted_id = torch.argmax(
            outputs.logits,
            dim=-1
        ).item()

    intent = LABEL_NAMES[predicted_id]

    # ==========================================
    # MODEL 2 PROMPT
    # ==========================================

    model_2_prompt = f"""Farmer query:
{user_message}

Farm context:
{farm_context_text}

Decision engine selected action:
{intent}

Generate a clear and helpful response for the farmer.

Rules:
- The selected action is FINAL.
- Do not change the selected action.
- Do not override the decision.
- Explain why the action was selected using the farm context.
- Keep the response practical and easy to understand.
"""

    # ==========================================
    # STREAM MODEL 2
    # ==========================================

    def generate():

        complete_response = ""

        for token in generate_response(
            model_2_prompt,
            max_new_tokens=200
        ):

            complete_response += token

            yield token

        # ======================================
        # SAVE COMPLETE RESPONSE TO MONGODB
        # ======================================

        now = datetime.utcnow()

        user_msg_doc = {
            'sender': 'user',
            'text': user_message,
            'timestamp': now
        }

        bot_msg_doc = {
            'sender': 'bot',
            'text': complete_response,
            'intent': intent,
            'timestamp': now
        }

        messages = chat_doc.get(
            'messages',
            []
        )

        if len(messages) == 0:

            title = ' '.join(
                user_message.split()
            )

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
                'user_id': user_id
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

    return Response(
        stream_with_context(generate()),
        mimetype='text/plain',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


def generate_response(user_prompt, max_new_tokens=200):

    messages = [
        {
            "role": "system",
            "content": (
                "You are Kaarshika's conversational response generator. "
                "The decision engine has already selected the action. "
                "Explain the selected action clearly without changing it."
            )
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    prompt = tokenizer_2.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer_2(
        prompt,
        return_tensors="pt"
    ).to("cuda")

    input_device = model_2.get_input_embeddings().weight.device

    inputs = {
        k: v.to(input_device)
        for k, v in inputs.items()
    }

    streamer = TextIteratorStreamer(
        tokenizer_2,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "eos_token_id": tokenizer_2.eos_token_id,
        "pad_token_id": tokenizer_2.eos_token_id
    }

    thread = Thread(
        target=model_2.generate,
        kwargs=generation_kwargs
    )

    thread.start()

    for new_text in streamer:
        yield new_text

    thread.join()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
