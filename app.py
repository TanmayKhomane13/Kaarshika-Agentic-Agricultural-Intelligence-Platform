from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch

app = Flask(__name__)

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

@app.route('/login')
def login():
    return render_template("login.html")

@app.route('/register')
def register():
    return render_template("register.html")

@app.route('/chat')
def chat():
    return render_template("chat.html")


@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    user_message = data.get('message', '').strip()

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

    return jsonify({
        'reply': intent          # Only the class name will be shown
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)