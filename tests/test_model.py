import numpy as np
import pandas as pd

from src.house_price_assistant import HousePriceAssistant


def test_prediction_returns_a_finite_positive_price():
    assistant = HousePriceAssistant()

    features = {
        "Gr Liv Area": 1800,
        "Overall Qual": 7,
    }

    prediction = assistant.predict(features)

    assert isinstance(prediction, (int, float, np.integer, np.floating))
    assert np.isfinite(prediction)
    assert prediction > 0


def test_prediction_uses_all_model_features_in_expected_order():
    assistant = HousePriceAssistant()

    features = {
        "Gr Liv Area": 1800,
        "Overall Qual": 7,
    }

    captured = {}
    original_predict = assistant.model.predict

    def recording_predict(X):
        captured["columns"] = list(X.columns)
        captured["shape"] = X.shape
        return original_predict(X)

    assistant.model.predict = recording_predict
    try:
        assistant.predict(features)
    finally:
        assistant.model.predict = original_predict

    assert captured["columns"] == assistant.feature_columns
    assert captured["shape"] == (1, len(assistant.feature_columns))


def test_model_meets_documented_test_set_performance():
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    import json
    from pathlib import Path

    from src.data_preprocessing import load_data, split_features_target

    root = Path(__file__).resolve().parents[1]

    metadata = json.loads(
        (root / "models" / "model_metadata.json").read_text(
            encoding="utf-8"
        )
    )

    df = load_data(root / "data" / "raw" / "ames_housing.csv")
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    assistant = HousePriceAssistant()
    predictions = assistant.model.predict(X_test[metadata["feature_columns"]])

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    assert len(X_train) == metadata["training_rows"]
    assert len(X_test) == metadata["test_rows"]
    assert rmse < 30000, f"Test RMSE exceeded $30,000: ${rmse:,.2f}"
    assert r2 > 0.88, f"Test R-squared was below 0.88: {r2:.4f}"
