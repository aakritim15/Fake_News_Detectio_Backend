import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import torch
import torch.nn as nn
import re
import string

# Define your PyTorch model architecture
class FakeNewsClassifier(nn.Module):
    def __init__(self, input_dim):
        super(FakeNewsClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.fc(x)

# Text preprocessing helper
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

# Initialize Flask
app = Flask(__name__)
CORS(app)  # allow cross-origin requests

# Load models from the `models/` directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# Scikit-learn vectorizer
vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))

# Dictionary of sklearn models
models = {
    "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, "Logistic_Regression.pkl")),
    "Naive Bayes":       joblib.load(os.path.join(MODEL_DIR, "Naive_Bayes.pkl")),
    "Random Forest":     joblib.load(os.path.join(MODEL_DIR, "Random_Forest.pkl")),
    "Gradient Boosting": joblib.load(os.path.join(MODEL_DIR, "Gradient_Boosting.pkl")),
}

# Load the PyTorch model
input_dim = vectorizer.max_features
pytorch_model = FakeNewsClassifier(input_dim)

# Load state dict (map to CPU for compatibility)
pth_path = os.path.join(MODEL_DIR, "fake_news_model.pth")
pytorch_model.load_state_dict(torch.load(pth_path, map_location="cpu"))
pytorch_model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Preprocess and vectorize
    cleaned = preprocess_text(text)
    vec = vectorizer.transform([cleaned])

    # Get sklearn predictions
    sk_preds = {name: int(model.predict(vec)[0]) for name, model in models.items()}

    # (Optional) PyTorch prediction
    # tensor_input = torch.tensor(vec.toarray(), dtype=torch.float32)
    # pt_output = pytorch_model(tensor_input)
    # pt_pred = torch.argmax(pt_output, dim=1).item()
    # sk_preds["PyTorch Model"] = pt_pred

    # Map to labels
    response = {name: ("Real" if pred == 1 else "Fake") for name, pred in sk_preds.items()}
    return jsonify(response)

if __name__ == "__main__":
    # Get port from environment (Railway sets this)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
