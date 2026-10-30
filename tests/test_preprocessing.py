import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from scipy import sparse

from src.data_preprocessing import build_preprocessor


def sample_features():
    """Return a small dataset containing numeric and categorical features."""
    return pd.DataFrame(
        {
            "Lot Area": [1000.0, 1500.0, np.nan, 2500.0],
            "Overall Qual": [4, 6, 8, 10],
            "Alley": ["Grvl", None, "Pave", None],
            "Neighborhood": ["NAmes", "CollgCr", None, "NAmes"],
        }
    )


def test_preprocessing_handles_missing_values():
    X = sample_features()
    preprocessor = build_preprocessor(X)

    transformed = preprocessor.fit_transform(X)
    values = transformed.data if sparse.issparse(transformed) else transformed

    assert np.isfinite(values).all()


def test_categorical_features_are_encoded():
    X = sample_features()
    preprocessor = build_preprocessor(X)

    transformed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()

    assert transformed.shape[0] == len(X)
    assert "Alley_None" in feature_names
    assert any(name.startswith("Neighborhood_") for name in feature_names)


def test_numeric_features_are_standardized():
    X = sample_features()
    preprocessor = build_preprocessor(X)
    preprocessor.fit(X)

    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    numeric_pipeline = preprocessor.named_transformers_["numeric"]
    transformed_numeric = numeric_pipeline.transform(X[numeric_columns])

    assert np.isfinite(transformed_numeric).all()
    np.testing.assert_allclose(
        transformed_numeric.mean(axis=0),
        np.zeros(len(numeric_columns)),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        transformed_numeric.std(axis=0),
        np.ones(len(numeric_columns)),
        atol=1e-10,
    )


def test_preprocessing_does_not_mutate_input():
    X = sample_features()
    original = X.copy(deep=True)

    preprocessor = build_preprocessor(X)
    preprocessor.fit_transform(X)

    assert_frame_equal(X, original)
