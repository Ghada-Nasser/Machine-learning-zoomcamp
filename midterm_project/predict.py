
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


MODEL_DIR = Path("models")


def load_artifacts():
    """Load the trained model, DictVectorizer, and threshold."""

    with open(MODEL_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)

    with open(MODEL_DIR / "dv.pkl", "rb") as f:
        dv = pickle.load(f)

    with open(MODEL_DIR / "threshold.pkl", "rb") as f:
        threshold = pickle.load(f)

    return model, dv, threshold


def prepare_features(session: dict[str, Any]) -> pd.DataFrame:
    """Convert a new session into the expected feature format."""

    categorical_columns = [
        "Month",
        "OperatingSystems",
        "Browser",
        "Region",
        "TrafficType",
        "VisitorType",
        "Weekend",
    ]

    session = session.copy()

    for column in categorical_columns:
        session[column] = str(session[column])

    return pd.DataFrame([session])


def predict(session: dict[str, Any]) -> dict[str, Any]:
    """Predict whether a session will result in a purchase."""

    model, dv, threshold = load_artifacts()

    session_df = prepare_features(session)
    session_dict = session_df.to_dict(orient="records")

    X = dv.transform(session_dict)

    probability = model.predict_proba(X)[0]
    positive_index = list(model.classes_).index(True)

    purchase_probability = probability[positive_index]
    prediction = purchase_probability >= threshold

    return {
        "purchase_probability": float(purchase_probability),
        "prediction": bool(prediction),
    }
