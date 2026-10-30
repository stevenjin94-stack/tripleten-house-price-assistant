from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "SalePrice"
IDENTIFIER_COLUMNS = ["Order", "PID"]

# In these columns, a missing value generally means the feature is absent.
ABSENCE_COLUMNS = [
    "Pool QC",
    "Misc Feature",
    "Alley",
    "Fence",
    "Mas Vnr Type",
    "Fireplace Qu",
    "Garage Type",
    "Garage Finish",
    "Garage Qual",
    "Garage Cond",
    "Bsmt Qual",
    "Bsmt Cond",
    "Bsmt Exposure",
    "BsmtFin Type 1",
    "BsmtFin Type 2",
]


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the housing dataset and validate its basic structure."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Dataset is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Required target column '{TARGET_COLUMN}' is missing."
        )

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Target contains missing values.")

    if (df[TARGET_COLUMN] <= 0).any():
        raise ValueError("Target must contain positive sale prices.")

    return df


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features and target without modifying the input DataFrame."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    X = df.drop(
        columns=[TARGET_COLUMN, *IDENTIFIER_COLUMNS],
        errors="ignore",
    ).copy()

    y = df[TARGET_COLUMN].copy()

    if X.empty:
        raise ValueError("No feature columns remain after removing identifiers.")

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build leakage-safe numeric and categorical preprocessing."""
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    absence_columns = [
        column
        for column in categorical_columns
        if column in ABSENCE_COLUMNS
    ]
    other_categorical_columns = [
        column
        for column in categorical_columns
        if column not in absence_columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    absence_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(
                strategy="constant",
                fill_value="None",
                keep_empty_features=True,
            )),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(
                strategy="most_frequent",
                keep_empty_features=True,
            )),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []

    if numeric_columns:
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if absence_columns:
        transformers.append(
            ("absence_categories", absence_pipeline, absence_columns)
        )

    if other_categorical_columns:
        transformers.append(
            ("categorical", categorical_pipeline, other_categorical_columns)
        )

    if not transformers:
        raise ValueError("No supported feature columns were found.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
        sparse_threshold=0.0,
    )
