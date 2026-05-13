from flask import Flask, request, jsonify
import joblib
import re

# Load model + vectorizer
model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__ProjetJ__)

def clean_tweet(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    tweet = data.get("text", "")

    cleaned = clean_tweet(tweet)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]

    return jsonify({"sentiment": pred})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

