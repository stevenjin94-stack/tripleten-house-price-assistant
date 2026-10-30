
from pathlib import Path

import mlflow
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def main():
    """Compare the latest completed MLflow run for each configured model."""
    config_path = ROOT / "configs" / "config.yaml"

    with config_path.open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    experiment_name = config["mlflow"]["experiment_name"]
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        output_format="pandas",
    )

    if runs.empty:
        raise RuntimeError(
            f"No MLflow runs found for experiment '{experiment_name}'."
        )

    required_metrics = [
        "metrics.cv_rmse_mean",
        "metrics.cv_rmse_std",
        "metrics.test_mae",
        "metrics.test_rmse",
        "metrics.test_r2",
    ]

    missing_columns = [
        column for column in required_metrics if column not in runs.columns
    ]
    if missing_columns:
        raise RuntimeError(
            f"Missing expected MLflow metrics: {missing_columns}"
        )

    if "status" in runs.columns:
        runs = runs[runs["status"] == "FINISHED"].copy()

    model_column = "params.model_name"
    if model_column not in runs.columns:
        raise RuntimeError(
            "MLflow runs do not contain the 'params.model_name' parameter."
        )

    # Restrict the report to model configurations in the YAML file.
    configured_models = {
        model["name"] for model in config["training"]["models"]
    }
    runs = runs[runs[model_column].isin(configured_models)].copy()

    # Keep the newest completed run for each model configuration.
    if "start_time" in runs.columns:
        runs = runs.sort_values("start_time", ascending=False)
    else:
        raise RuntimeError(
            "MLflow runs do not contain start_time; cannot identify "
            "the latest run for each model."
        )

    latest_runs = runs.drop_duplicates(
        subset=[model_column],
        keep="first",
    ).copy()

    if latest_runs.empty:
        raise RuntimeError(
            "No completed runs found for the configured models."
        )

    # Require CV metrics and all three held-out test metrics.
    incomplete = latest_runs[
        latest_runs[required_metrics].isna().any(axis=1)
    ]

    if not incomplete.empty:
        problem_models = incomplete[model_column].tolist()
        raise RuntimeError(
            "The latest completed run is missing required metrics for "
            f"{problem_models}. Run 'python -m src.train' to log complete "
            "metrics for all configured models, then rerun this script."
        )

    # Lower CV RMSE is better. Test metrics are reported, not used to select.
    latest_runs = latest_runs.sort_values(
        "metrics.cv_rmse_mean",
        ascending=True,
    ).copy()

    display_columns = [
        column
        for column in [
            "run_id",
            "tags.mlflow.runName",
            "params.model_name",
            "params.estimator",
            "metrics.cv_rmse_mean",
            "metrics.cv_rmse_std",
            "metrics.test_mae",
            "metrics.test_rmse",
            "metrics.test_r2",
        ]
        if column in latest_runs.columns
    ]

    latest_runs["selected"] = False
    latest_runs.iloc[0, latest_runs.columns.get_loc("selected")] = True

    print("\n--- Latest MLflow Run per Model ---")
    print(latest_runs[display_columns + ["selected"]].to_string(index=False))

    best = latest_runs.iloc[0]

    print("\n--- Best Model Identified from MLflow ---")
    print(f"Model: {best[model_column]}")
    print(f"Run ID: {best['run_id']}")
    print(f"CV RMSE: ${best['metrics.cv_rmse_mean']:,.2f}")
    print(f"Test MAE: ${best['metrics.test_mae']:,.2f}")
    print(f"Test RMSE: ${best['metrics.test_rmse']:,.2f}")
    print(f"Test R²: {best['metrics.test_r2']:.4f}")

    report_path = ROOT / "reports" / "mlflow_experiment_comparison.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    latest_runs[display_columns + ["selected"]].to_csv(
        report_path,
        index=False,
    )
    print(f"\nComparison saved to: {report_path}")


if __name__ == "__main__":
    main()
