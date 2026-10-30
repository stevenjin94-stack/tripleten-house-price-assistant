# House Price Prediction Assistant



An end-to-end machine learning project that predicts residential sale prices using historical Ames, Iowa housing data. The application combines a trained scikit-learn model with Google's Gemini API to interpret natural-language property descriptions, extract supported features, generate price estimates, and explain the results.



## Project Overview



This project demonstrates a complete machine learning workflow, including data preprocessing, model comparison, experiment tracking, automated testing, and an LLM-powered user interface.



The assistant accepts a description of a property, extracts relevant features, passes them through a trained machine learning pipeline, and returns a price estimate with an explanation of the model's limitations.



**Important:** Predictions are statistical estimates based on historical sales data. They are not current market valuations, formal appraisals, or guarantees of sale prices.



## Results



The selected model is `gradient_boosting_baseline`, chosen based on its lowest mean cross-validation RMSE among the five evaluated configurations.



| Metric | Result |

|---|---:|

| Cross-validation RMSE | $25,877.94 |

| Test MAE | $16,242.10 |

| Test RMSE | $26,757.49 |

| Test R-squared | 0.9107 |


The held-out test results indicate that the model explains approximately 91% of the variance in sale prices in the test dataset. Its test MAE means that predictions differed from actual prices by about $16,242 on average in absolute terms.



### Model Comparison



| Model configuration | Mean CV RMSE |

|---|---:|

| Gradient Boosting Baseline | $25,877.94 |

| Random Forest Regularized | $26,603.83 |

| Random Forest Baseline | $26,639.32 |

| Extra Trees Baseline | $26,817.20 |

| Extra Trees Regularized | $27,004.91 |



Lower RMSE indicates better predictive performance. The Gradient Boosting Baseline configuration had the lowest mean cross-validation RMSE and was selected for the final model. All five model configurations were evaluated on the same held-out test set, and their test MAE, test RMSE, and R-squared metrics are included in `reports/mlflow_experiment_comparison.csv`. Model selection was based on cross-validation RMSE, not held-out test performance.


## Dataset



The project uses the Ames Housing dataset, containing 2,930 residential property records and 82 original columns. The prediction target is `SalePrice`.



\- **Rows:** 2,930

\- **Target:** `SalePrice`

\- **Original columns:** 82

\- **Model input features:** 79, excluding the target and the `Order` and `PID` identifier columns



The preprocessing pipeline handles missing numeric and categorical values, scales numeric features, and one-hot encodes categorical features. The fitted preprocessing steps are part of the saved model pipeline.



### Dataset Source and Attribution

The Ames Housing dataset was compiled by Dean De Cock and describes residential property sales in Ames, Iowa, from 2006 through 2010. The original dataset contains 2,930 observations and 82 variables.

- **Original research paper:** De Cock, D. (2011). *Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project.* Journal of Statistics Education, 19(3).
- **Paper and dataset documentation:** https://jse.amstat.org/v19n3/decock.pdf
- **Variable documentation:** https://jse.amstat.org/v19n3/decock/DataDocumentation.txt

The data originated from the Ames Assessor's Office. Confirm the applicable terms and permissions for the specific copy of the dataset used in this project before redistributing it. This project does not claim that the original dataset is unrestricted or that it has a particular open-source license.



## Architecture



1\. **Data preparation:** Load and validate the housing dataset, separate features and target, and prepare preprocessing transformations.

2\. **Model training:** Train five configurations defined in `configs/config.yaml`.

3\. **Experiment tracking:** Log model parameters, cross-validation results, and final evaluation metrics with MLflow.

4\. **Model selection:** Compare runs and select the configuration with the lowest mean cross-validation RMSE.

5\. **Prediction assistant:** Use Gemini to extract explicitly stated property features from natural language.

6\. **Price prediction:** Pass the extracted features to the saved scikit-learn pipeline, with missing inputs handled by preprocessing.

7\. **Explanation:** Ask Gemini to explain the estimate and its limitations.

8\. **Testing:** Use pytest to check preprocessing, prediction behavior, JSON parsing, and incomplete or invalid assistant inputs.


## Repository Structure

tripleten-capstone/
|-- configs/
|   `-- config.yaml
|-- data/
|   |-- raw/
|   `-- processed/
|-- docs/
|   `-- demo/
|       `-- house_price_assistant_demo.mp4
|-- models/
|-- notebooks/
|-- reports/
|   |-- mlflow_experiment_comparison.csv
|   `-- model_comparison.csv
|-- src/
|   |-- compare_experiments.py
|   |-- data_preprocessing.py
|   |-- house_price_assistant.py
|   `-- train.py
|-- tests/
|   |-- test_assistant.py
|   |-- test_model.py
|   `-- test_preprocessing.py
|-- .env.example
|-- .gitignore
|-- README.md
`-- requirements.txt
```

The raw dataset, trained model artifacts, local environment file, virtual environment, and MLflow database are excluded from Git. The report CSV files are included.
## Requirements



\- Python 3.11

\- Google AI Studio API key

\- Dependencies listed in `requirements.txt`



## Setup



### 1. Create and activate a virtual environment



On Windows PowerShell:



```powershell

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

```



### 2. Install dependencies



```powershell

python -m pip install -r requirements.txt

```



## Run Tests

Before running the test suite for the first time in a fresh clone, complete the dataset preparation steps in the Verify the dataset and model artifacts section and train the models using:


```powershell
python -m src.train
```

This creates the trained model artifacts required by the model and assistant tests.

Run the automated test suite from the repository root with the project virtual environment activated:

```powershell
python -m pytest tests/ -v
```

The current test suite contains 12 tests covering:

- **Preprocessing (4 tests):** handling missing values, categorical encoding, numeric standardization, and preserving the original input data.
- **Model (3 tests):** valid positive predictions, correct prediction feature ordering and input shape, and held-out test-set performance.
- **Assistant interface (5 tests):** JSON parsing, invalid JSON handling, rejection of unknown features, requests for missing required details, and successful natural-language feature extraction.

All 12 tests passed during the final local verification.

The Gemini integration was also manually tested end to end. Automated tests use controlled responses where appropriate to avoid requiring live API calls for every test.


## Verify the dataset and model artifacts

Before running the assistant, make sure the Ames Housing dataset and trained model artifacts are available at the expected paths. These files are excluded from Git to keep the repository lightweight and avoid committing data or generated model files.

### Obtain and prepare the dataset

1. Download the original tab-delimited Ames Housing dataset from the source published with the original research:

   https://jse.amstat.org/v19n3/decock/AmesHousing.txt

2. Save the downloaded file as:

   `data/raw/AmesHousing.txt`

3. Activate your project virtual environment, install the project dependencies, and run the following command from the project root in PowerShell:

   ```powershell
   python -c "import pandas as pd; df = pd.read_csv(r'data/raw/AmesHousing.txt', sep='\t'); assert df.shape == (2930, 82), f'Unexpected dataset shape: {df.shape}'; assert 'SalePrice' in df.columns, 'SalePrice target column is missing'; df.to_csv(r'data/raw/ames_housing.csv', index=False); print('Dataset prepared:', df.shape)"
   ```

4. Confirm that the prepared dataset exists at:

   `data/raw/ames_housing.csv`

   The command checks the expected dataset dimensions and verifies that the `SalePrice` target column is present before creating the CSV used by this project. If a check fails, stop and investigate the downloaded file rather than proceeding with an unverified dataset.

Check the dataset's applicable terms of use before redistributing it. The dataset is not included in this Git repository.

### Verify the trained model artifacts

The assistant requires these two files:

- `models/best_model.joblib` — the trained model pipeline.
- `models/model_metadata.json` — model metadata, including the expected feature information.

If either file is missing, train the models from the project root:

```powershell
python -m src.train
```

After training completes, confirm that both files exist before running the assistant.

You can check the expected paths in PowerShell with:

```powershell
Test-Path .\data\raw\ames_housing.csv
Test-Path .\models\best_model.joblib
Test-Path .\models\model_metadata.json
```

Each command should return `True` before you proceed. If a result is `False`, complete the relevant dataset preparation or model-training step first.

## Run the Assistant



From the repository root:



```powershell

python -m src.house_price_assistant

```



Describe a property in natural language. For example:



> Estimate the price of a house with 1,800 square feet of above-ground living area and an overall quality rating of 7 out of 10.



The assistant extracts the explicitly provided features, generates a prediction, and provides an explanation. If required details are missing, it asks for clarification.



When only some model inputs are provided, the preprocessing pipeline imputes the remaining missing values. Estimates based on incomplete descriptions should therefore be interpreted cautiously.



## Train and Compare Models



To run the configured training workflow:



```powershell

python -m src.train

```



To compare recorded experiments:



```powershell

python -m src.compare_experiments

```



Training configurations and the MLflow tracking URI are defined in `configs/config.yaml`. Experiment results are stored in the configured MLflow tracking store, and comparison reports are written to `reports/`.


### Inspect MLflow experiments

The training workflow records five model experiment runs in MLflow. Each run includes the model configuration, hyperparameters, cross-validation RMSE, held-out test metrics, and trained model artifact.

To open the MLflow user interface from the project root, run:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open:

http://127.0.0.1:5000

The experiment is named `house-price-model-comparison`. Use the MLflow interface to inspect individual runs, compare parameters and metrics, and review logged artifacts.

The programmatic comparison script also uses `mlflow.search_runs()` to identify the best completed model run based on cross-validation RMSE.



Training can take time because the workflow evaluates multiple configurations using cross-validation. The existing model artifacts and MLflow database can be reused when available.





## Limitations and Future Improvements



\- The model is trained on historical Ames, Iowa sales and may not generalize to other regions or current market conditions.

\- Incomplete property descriptions require imputation, which can reduce estimate reliability.

\- LLM-generated feature extraction and explanations can be imperfect and should be validated.

\- Future work could add confidence intervals, stronger validation for ambiguous property descriptions, a web interface, and deployment with Docker.

\- Model performance should be monitored if new data becomes available.


## Project Demonstration

The recorded demonstration is available in the repository:

[Watch the House Price Prediction Assistant Demo](docs/demo/house_price_assistant_demo.mp4)

The video demonstrates:

- A normal natural-language house-price request.
- An incomplete property description and the assistant's clarification response.
- The predicted house price and AI-generated explanation.


## Project Reflection

Building this project helped me understand that a successful machine learning solution involves much more than training a model. Data preprocessing, preventing data leakage, comparing model configurations, tracking experiments, and evaluating performance on a held-out test set are all important parts of a reproducible workflow.

Integrating a large language model with the trained pipeline also highlighted the importance of validating extracted features and asking follow-up questions when essential property information is missing. The language model provides the natural-language interface, while the trained machine learning pipeline generates the numerical estimate.

One of the most important lessons was the need to communicate model limitations clearly. This project uses historical home sales from Ames, Iowa, so its estimates should not be treated as current market valuations or assumed to generalize to other locations.

If I continued developing the project, I would add a more user-friendly web interface, improve uncertainty estimates, evaluate performance on additional housing datasets, and explore a deployment workflow using Docker. I would also monitor model performance as new data becomes available.



## Reproducibility and Security



\- Training settings are stored in `configs/config.yaml`.

\- MLflow tracks experiments and evaluation metrics.

\- Automated checks run with pytest.

\- API credentials belong in a local .env file, not in Git.

\- Dataset redistribution and licensing requirements should be checked before publishing data files.



Author



Capstone project demonstrating machine learning, experiment tracking, automated testing, and LLM integration for residential price estimation.
