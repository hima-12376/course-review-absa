from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sentiment_model import SentimentAnalyzer
from aspect_extractor import extract_aspects

app = Flask(__name__)
CORS(app)

# 🔥 Load model once when server starts
print("Loading sentiment model...")
model = SentimentAnalyzer()
print("Model loaded successfully!")

# -------------------------------------
# 🏠 Home Route (Serves Frontend)
# -------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -------------------------------------
# 🤖 Analyze Route (API)
# -------------------------------------
@app.route("/analyze", methods=["POST"])
def analyze():

    data = request.get_json()

    if not data or "review" not in data:
        return jsonify({"error": "No review provided"}), 400

    review = data["review"].strip()

    if review == "":
        return jsonify({"error": "Empty review"}), 400

    # Extract aspects using spaCy
    aspects = extract_aspects(review)

    if not aspects:
        return jsonify([])

    results = []

    for aspect in aspects:
        sentiment, confidence = model.predict_aspect_sentiment(review, aspect)

        # Convert sentiment to status for UI colors
        if sentiment == "Positive":
            status = "positive"
        elif sentiment == "Negative":
            status = "negative"
        else:
            status = "neutral"

        results.append({
            "aspect": aspect,
            "sentiment": confidence,   # percentage value
            "status": status,
            "description": f"Detected {sentiment} sentiment with {confidence}% confidence."
        })

    return jsonify(results)


# -------------------------------------
# 🚀 Run Server
# -------------------------------------
if __name__ == "__main__":
    app.run(debug=True)