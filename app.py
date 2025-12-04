from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib
import numpy as np
import os
import io
from datetime import datetime, timezone
# Optional: reportlab is imported lazily inside generate_report to avoid runtime import errors when PDF generation isn't used.

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "model.pkl"))

app = Flask(__name__)
CORS(app)

# Load model at startup
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    app.logger.warning(f"Could not load model at {MODEL_PATH}: {e}")


def bucketize(prob: float) -> str:
    # Convert probability of positive class into Low/Medium/High buckets
    if prob < 0.33:
        return "Low"
    elif prob < 0.66:
        return "Medium"
    else:
        return "High"


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Train first."}), 500

    data = request.get_json(silent=True) or {}
    missing = [k for k in ["bp", "heart_rate", "sugar", "bmi"] if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        bp = float(data["bp"])           # systolic mmHg
        hr = float(data["heart_rate"])   # bpm
        sugar = float(data["sugar"])     # mg/dL
        bmi = float(data["bmi"])         # kg/m^2
    except (TypeError, ValueError):
        return jsonify({"error": "All fields must be numbers"}), 400

    X = np.array([[bp, hr, sugar, bmi]], dtype=float)
    try:
        proba = float(model.predict_proba(X)[0, 1])
        risk = bucketize(proba)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    return jsonify({"risk": risk, "probability": round(proba, 4)})


@app.route("/api/generate-report", methods=["POST"])
def generate_report():
    """
    Accepts JSON payload:
    {
      "user": { "name": str, "email": str, ... },
      "readings": [ { "date": str, "systolic": number, "diastolic": number, "heartRate": number, "sugar": number, "bmi": number }, ... ],
      "prediction": { "risk": str, "probability": float },
      "ai_summary": str (optional)
    }
    Returns: application/pdf
    """
    data = request.get_json(silent=True) or {}
    # Lazy import reportlab so the API can run without it for prediction-only usage.
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
    except Exception as e:
        return jsonify({"error": f"PDF generation unavailable: {e}"}), 501

    try:
        buf = io.BytesIO()
        _build_pdf(buf, data)
        buf.seek(0)
    except Exception as e:
        return jsonify({"error": f"Failed to generate PDF: {e}"}), 500

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    filename = f"HealTrack_Report_{ts}.pdf"
    return send_file(buf, mimetype="application/pdf", as_attachment=True, download_name=filename)


def _build_pdf(buffer: io.BytesIO, data: dict):
    # Import here as well to keep module import lightweight
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    elements = []

    # Header
    elements.append(Paragraph("HealTrack Health Report", styles['Title']))
    elements.append(Spacer(1, 12))
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    elements.append(Paragraph(f"Generated: {ts}", styles['Normal']))
    elements.append(Spacer(1, 12))

    # User details
    user = data.get('user') or {}
    elements.append(Paragraph("User Details", styles['Heading2']))
    user_lines = []
    for k in ["name", "email", "age", "gender", "id"]:
        if user.get(k) is not None:
            user_lines.append(f"<b>{k.title()}</b>: {user.get(k)}")
    if not user_lines:
        user_lines = ["No user details provided"]
    for line in user_lines:
        elements.append(Paragraph(line, styles['Normal']))
    elements.append(Spacer(1, 12))

    # Readings table
    elements.append(Paragraph("Last 5 Health Readings", styles['Heading2']))
    readings = (data.get('readings') or [])[:5]
    table_data = [["Date", "Systolic", "Diastolic", "Heart Rate", "Glucose", "BMI"]]
    for r in readings:
        table_data.append([
            str(r.get('date', '')),
            str(r.get('systolic', '')),
            str(r.get('diastolic', '')),
            str(r.get('heartRate', '')),
            str(r.get('sugar', '')),
            str(r.get('bmi', '')),
        ])
    table = Table(table_data, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f3f4f6')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#111827')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e5e7eb')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN', (1,1), (-1,-1), 'CENTER'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#fafafa')]),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))

    # AI summary
    elements.append(Paragraph("AI Prediction Summary", styles['Heading2']))
    ai_summary = data.get('ai_summary')
    if not ai_summary:
        pred = data.get('prediction') or {}
        risk = pred.get('risk')
        proba = pred.get('probability')
        if risk is not None and proba is not None:
            ai_summary = f"Risk: {risk} (probability {proba})"
    elements.append(Paragraph(ai_summary or 'N/A', styles['Normal']))

    doc.build(elements)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
