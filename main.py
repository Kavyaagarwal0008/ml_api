from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow all origins for local development


def _to_float(x, name):
    try:
        return float(x)
    except (TypeError, ValueError):
        raise ValueError(f"'{name}' must be a number")


def compute_risk(bp: float, heart_rate: float, sugar: float, bmi: float):
    """
    Simple transparent rule-based risk scoring.
    Returns (label, probability in [0,1]).
    """
    score = 0.0

    # Blood pressure (systolic)
    if bp >= 180:
        score += 3
    elif bp >= 160:
        score += 2.5
    elif bp >= 140:
        score += 2
    elif bp <= 90:
        score += 0.5

    # Heart rate
    if heart_rate >= 120:
        score += 2
    elif heart_rate >= 100:
        score += 1.5
    elif heart_rate <= 50:
        score += 1

    # Fasting glucose (mg/dL)
    if sugar >= 250:
        score += 3
    elif sugar >= 180:
        score += 2
    elif sugar >= 126:
        score += 1

    # BMI
    if bmi >= 35:
        score += 2.5
    elif bmi >= 30:
        score += 2
    elif bmi <= 18.5:
        score += 1

    # Map score to risk band
    if score <= 2.0:
        label = "Low"
        prob = min(0.35, 0.15 + 0.1 * score)
    elif score <= 4.0:
        label = "Medium"
        prob = min(0.7, 0.45 + 0.1 * (score - 2))
    else:
        label = "High"
        prob = min(0.95, 0.7 + 0.05 * (score - 4))

    # Clamp probability
    prob = max(0.01, min(0.99, prob))
    return label, prob


@app.route("/predict", methods=["POST"])  # expected by src/mlApi.js
def predict():
    try:
        payload = request.get_json(force=True, silent=False)
        if not isinstance(payload, dict):
            return jsonify({"error": "Invalid JSON body"}), 400

        # Accept keys exactly as frontend sends them
        bp = _to_float(payload.get("bp"), "bp")
        heart_rate = _to_float(payload.get("heart_rate"), "heart_rate")
        sugar = _to_float(payload.get("sugar"), "sugar")
        bmi = _to_float(payload.get("bmi"), "bmi")

        risk, probability = compute_risk(bp, heart_rate, sugar, bmi)
        return jsonify({
            "risk": risk,                  # 'Low' | 'Medium' | 'High'
            "probability": round(probability, 3),
            "inputs": {"bp": bp, "heart_rate": heart_rate, "sugar": sugar, "bmi": bmi},
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # Avoid leaking internals; still log for dev
        app.logger.exception("Prediction failed: %s", e)
        return jsonify({"error": "Prediction failed"}), 500


@app.route("/health", methods=["GET"])  # simple health check
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    import os

    port = int(os.getenv("PORT", "5001"))
    # Host 0.0.0.0 so browser on same machine can reach it; debug for hot reload in dev
    app.run(host="0.0.0.0", port=port, debug=True)
