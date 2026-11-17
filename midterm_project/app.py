
from flask import Flask, request, jsonify

from predict import predict

app = Flask("online-shoppers-prediction")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Welcome to Online Shoppers Purchase Prediction API"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok"
    })


@app.route("/predict", methods=["POST"])
def predict_endpoint():

    session = request.get_json()

    result = predict(session)

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
