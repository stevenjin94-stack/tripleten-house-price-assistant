import hashlib
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.data_preprocessing import (
    build_preprocessor,
    load_data,
    split_features_target,
)


ROOT = Path(__file__).resolve().parents[1]


def load_config():
    """Load project configuration from YAML."""
    config_path = ROOT / "configs" / "config.yaml"
    with config_path.open("r", encoding="utf-8-sig") as file:
        return yaml.safe_load(file)


def sha256_file(path):
    """Calculate a reproducible SHA-256 hash of a dataset file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model(model_config, random_state):
    """Create a supported estimator from its YAML configuration."""
    model_params = {
        key: value
        for key, value in model_config.items()
        if key not in {"name", "estimator"}
    }
    model_params["random_state"] = random_state

    estimator_name = model_config["estimator"]

    estimators = {
        "random_forest": RandomForestRegressor,
        "extra_trees": ExtraTreesRegressor,
        "gradient_boosting": GradientBoostingRegressor,
    }

    if estimator_name not in estimators:
        raise ValueError(f"Unsupported estimator: {estimator_name}")

    return estimators[estimator_name](**model_params)


def build_pipeline(model_config, X_train, random_state):
    """Keep preprocessing and model training together to prevent leakage."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("model", build_model(model_config, random_state)),
        ]
    )


def main():
    config = load_config()

    data_path = (ROOT / config["data"]["raw_path"]).resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = load_data(data_path)
    X, y = split_features_target(df)

    if config["data"]["target_column"] != "SalePrice":
        raise ValueError(
            "Configuration target must match the preprocessing module's "
            "target column, SalePrice."
        )

    random_state = config["project"]["random_state"]
    test_size = config["data"]["test_size"]
    folds = config["training"]["cross_validation_folds"]
    selection_metric = config["training"]["selection_metric"]

    if selection_metric != "cv_rmse":
        raise ValueError(
            f"Unsupported selection metric: {selection_metric}. "
            "Supported metric: cv_rmse."
        )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    dataset_hash = sha256_file(data_path)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    cv = KFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_state,
    )

    run_results = []
    best_pipeline = None
    best_run_id = None
    best_model_config = None
    best_cv_rmse = float("inf")
    best_cv_std = None

    print(f"Dataset: {data_path.name}")
    print(f"Dataset rows: {len(df)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Input features: {X.shape[1]}")
    print(f"Dataset SHA-256: {dataset_hash}")
    print(f"Cross-validation folds: {folds}")
    print(f"Experiments configured: {len(config['training']['models'])}")


    for model_config in config["training"]["models"]:
        run_name = model_config["name"]
        print(f"\nStarting experiment: {run_name}")

        pipeline = build_pipeline(model_config, X_train, random_state)

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(
                {
                    "model_name": run_name,
                    "estimator": model_config["estimator"],
                    "target_column": config["data"]["target_column"],
                    "dataset_filename": data_path.name,
                    "dataset_sha256": dataset_hash,
                    "dataset_rows": len(df),
                    "input_feature_count": X.shape[1],
                    "test_size": test_size,
                    "random_state": random_state,
                    "cv_folds": folds,
                    **{
                        f"model_{key}": value
                        for key, value in model_config.items()
                        if key not in {"name", "estimator"}
                    },
                }
            )

            scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                scoring="neg_root_mean_squared_error",
                cv=cv,
                n_jobs=1,
                error_score="raise",
            )

            cv_rmse = -scores
            mean_cv_rmse = float(np.mean(cv_rmse))
            std_cv_rmse = float(np.std(cv_rmse, ddof=1))

            mlflow.log_metric("cv_rmse_mean", mean_cv_rmse)
            mlflow.log_metric("cv_rmse_std", std_cv_rmse)

            # Refit this candidate on training data only.
            pipeline.fit(X_train, y_train)

            # Evaluate each candidate on the same held-out test set.
            test_predictions = pipeline.predict(X_test)

            candidate_test_metrics = {
                "test_mae": float(
                    mean_absolute_error(y_test, test_predictions)
                ),
                "test_rmse": float(
                    np.sqrt(mean_squared_error(y_test, test_predictions))
                ),
                "test_r2": float(r2_score(y_test, test_predictions)),
            }

            mlflow.log_metrics(candidate_test_metrics)

            # Log the fitted candidate model to this MLflow run.
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
            )

            run_id = run.info.run_id

        run_results.append(
            {
                "run_id": run_id,
                "model_name": run_name,
                "estimator": model_config["estimator"],
                "cv_rmse_mean": mean_cv_rmse,
                "cv_rmse_std": std_cv_rmse,
                **candidate_test_metrics,
            }
        )

        print(f"Mean CV RMSE: ${mean_cv_rmse:,.2f}")
        print(f"CV RMSE standard deviation: ${std_cv_rmse:,.2f}")
        print(f"Test MAE: ${candidate_test_metrics['test_mae']:,.2f}")
        print(f"Test RMSE: ${candidate_test_metrics['test_rmse']:,.2f}")
        print(f"Test R-squared: {candidate_test_metrics['test_r2']:.4f}")

        # Select the winner using CV RMSE, not test metrics.
        if mean_cv_rmse < best_cv_rmse:
            best_cv_rmse = mean_cv_rmse
            best_cv_std = std_cv_rmse
            best_pipeline = pipeline
            best_run_id = run_id
            best_model_config = model_config.copy()

    if best_pipeline is None:
        raise RuntimeError("No model was successfully trained.")


    # Build the complete comparison results before creating metadata.
    results_df = pd.DataFrame(run_results)
    results_df["selected"] = results_df["run_id"].eq(best_run_id)

    # Extract the held-out metrics for the selected model.
    best_test_row = results_df.loc[
        results_df["run_id"].eq(best_run_id)
    ].iloc[0]

    best_test_metrics = {
        "test_mae": float(best_test_row["test_mae"]),
        "test_rmse": float(best_test_row["test_rmse"]),
        "test_r2": float(best_test_row["test_r2"]),
    }



    models_dir = ROOT / "models"
    reports_dir = ROOT / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "best_model.joblib"
    metadata_path = models_dir / "model_metadata.json"
    report_path = reports_dir / "model_comparison.csv"

    joblib.dump(best_pipeline, model_path)

    metadata = {
        "project": config["project"]["name"],
        "model_name": best_model_config["name"],
        "estimator": best_model_config["estimator"],
        "model_parameters": best_model_config,
        "target_column": config["data"]["target_column"],
        "feature_columns": X.columns.tolist(),
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "dataset_filename": data_path.name,
        "dataset_sha256": dataset_hash,
        "cv_folds": folds,
        "cv_rmse_mean": best_cv_rmse,
        "cv_rmse_std": best_cv_std,
	"test_metrics": best_test_metrics,
        "mlflow_run_id": best_run_id,
        "interpretation": (
            "Prediction estimates historical sale price patterns; "
            "it is not a guaranteed current market valuation."
        ),
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


    results_df.sort_values("cv_rmse_mean").to_csv(
        report_path,
        index=False,
    )

    # Attach metadata and comparison report to the winning MLflow run.
    with mlflow.start_run(run_id=best_run_id):
        mlflow.log_artifact(str(metadata_path))
        mlflow.log_artifact(str(report_path))


    print("\n--- FINAL MODEL SELECTION ---")
    print(f"Selected model: {best_model_config['name']}")
    print(f"Winning MLflow run: {best_run_id}")
    print(f"Mean CV RMSE: ${best_cv_rmse:,.2f}")
    print("\n--- HELD-OUT TEST METRICS ---")
    print(f"MAE:  ${best_test_metrics['test_mae']:,.2f}")
    print(f"RMSE: ${best_test_metrics['test_rmse']:,.2f}")
    print(f"R-squared:   {best_test_metrics['test_r2']:.4f}")


if __name__ == "__main__":
    main()
