import os
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# We synthesize a small, plausible dataset for demo purposes.
# Features: [bp (systolic mmHg), heart_rate (bpm), sugar (mg/dL), bmi]
# Target: risk (0/1) derived from a simple heuristic with noise.

def generate_sample(n=1500, seed=42):
    rng = np.random.default_rng(seed)
    bp = rng.normal(125, 20, n).clip(80, 200)
    hr = rng.normal(75, 15, n).clip(45, 180)
    sugar = rng.normal(110, 35, n).clip(60, 320)
    bmi = rng.normal(26, 5, n).clip(16, 48)

    # Heuristic risk score: positive weights for higher-than-ideal values
    score = (
        0.03 * (bp - 120) +
        0.04 * (hr - 70) +
        0.02 * (sugar - 100) +
        0.06 * (bmi - 24)
    )
    noise = rng.normal(0, 0.5, n)
    y = (score + noise > 1.2).astype(int)  # roughly 30-40% positives

    X = np.column_stack([bp, hr, sugar, bmi])
    return X, y


def train_and_save(model_path="model.pkl"):
    X, y = generate_sample()

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, solver="lbfgs")),
    ])
    pipe.fit(X, y)
    joblib.dump(pipe, model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    out = os.environ.get("MODEL_PATH", "model.pkl")
    train_and_save(out)
