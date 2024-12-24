# Production Ready End-to-End House Price Prediction System 

Welcome to the **House Price Prediction** project! This end-to-end machine learning (ML) system is designed to predict house prices accurately using a robust, reusable, and production-ready pipeline. Built using **ZenML**, it incorporates state-of-the-art design patterns to enhance readability, modularity, and scalability.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Design Patterns Used](#design-patterns-used)
- [Pipeline Architecture](#pipeline-architecture)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Getting Started](#getting-started)
- [Steps in the Pipeline](#steps-in-the-pipeline)
- [Deployment](#deployment)
- [License](#license)

---

## Overview

This project focuses on creating a machine learning pipeline that predicts house prices based on various features such as square footage, location attributes, and property details. Using **ZenML**, this pipeline streamlines the ML lifecycle, from data ingestion to model deployment and inference.

**Key Objectives:**

- Perform data preprocessing, feature engineering, and outlier handling.
- Conduct in-depth Exploratory Data Analysis (EDA).
- Train and evaluate regression models.
- Enable reusable, modular pipelines for scalability and maintainability.
- Support continuous deployment for production-readiness.

---

## Features

- **Modular Design:** Each step in the pipeline is implemented as a reusable ZenML step.
- **Dynamic Data Handling:** Automatically handle missing values, outliers, and feature transformations.
- **EDA Insights:** Detailed data inspection, univariate, bivariate, and multivariate analysis.
- **Continuous Deployment:** Integrated with MLFlow for model deployment and monitoring.
- **Extensibility:** Supports customization of feature engineering strategies and model configurations.

---

## Design Patterns Used

1. **Singleton Design Pattern**

   - Ensures a single instance of critical components (e.g., experiment tracker, data loader) is used throughout the pipeline.

2. **Factory Design Pattern**

   - Simplifies data ingestion by dynamically selecting appropriate ingestors based on file formats.

3. **Template Design Pattern**

   - Standardizes preprocessing and feature engineering steps for consistent and reusable logic.

---

## Pipeline Architecture

The pipeline consists of the following key steps:

1. **Data Ingestion:** Import raw data from ZIP files or other formats.
2. **Missing Value Handling:** Fill or drop missing data using customizable strategies.
3. **Feature Engineering:** Apply transformations like logarithmic scaling, one-hot encoding, and standard scaling.
4. **Outlier Detection:** Identify and remove outliers based on Z-score or other criteria.
5. **Data Splitting:** Divide data into training and testing sets.
6. **Model Training:** Train a regression model using a scikit-learn pipeline.
7. **Model Evaluation:** Compute metrics such as Mean Squared Error (MSE).
8. **Model Deployment:** Deploy the model using MLFlow for inference and monitoring.

---

## Exploratory Data Analysis (EDA)

### 1. Basic Data Inspection

- **Data Overview:** The dataset contains **2930 entries** and **82 columns**, including both numerical and categorical features.
- **Data Types:**
  - 11 columns with `float64` type.
  - 28 columns with `int64` type.
  - 43 columns with `object` type.
- **Key Observations:**
  - Numerical features like `Gr Liv Area`, `Lot Area`, and `Year Built` vary significantly, indicating potential outliers.
  - Categorical features such as `Neighborhood` and `MS Zoning` have multiple unique values, influencing model encoding strategies.

### 2. Missing Values Analysis

- **Columns with Significant Missing Values:**
  - `Alley` (93.4% missing), `Pool QC` (99.5% missing), and `Fence` (80.5% missing).
- **Columns with Moderate Missing Values:**
  - `Garage Type` (5.4% missing) and `Bsmt Qual` (2.7% missing).
- **Insights:**
  - Severe missingness in some features suggests dropping or imputing with placeholders (e.g., "None").
  - Minor missing values can be imputed with mean, median, or mode.

### 3. Univariate Analysis

- **`SalePrice`:** Positively skewed with most prices between $100,000 and $250,000. Applying a log transformation can normalize the distribution.
- **`Neighborhood`:** `NAmes` is the most frequent, with some neighborhoods having sparse data.

### 4. Bivariate Analysis

- **`Gr Liv Area` vs. `SalePrice`:** Strong positive correlation (~0.71), suggesting living area is a key predictor.
- **`Overall Qual` vs. `SalePrice`:** Clear positive relationship, with higher quality leading to higher prices.

### 5. Multivariate Analysis

- **Correlation Insights:**
  - `SalePrice` correlates strongly with `Overall Qual` (0.80) and `Gr Liv Area` (0.71).
  - Potential multicollinearity among predictors like `Total Bsmt SF` and `Gr Liv Area`.
- **Key Observations:**
  - `Overall Qual` and `Gr Liv Area` are critical predictors.
  - Feature engineering (e.g., combining related features) can capture additional variance.

---

## Getting Started

### Prerequisites

- Python >= 3.8
- ZenML >= 0.20.0
- MLFlow >= 1.28.0
- scikit-learn >= 1.0.0

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/house-price-prediction.git
   cd house-price-prediction
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Initialize ZenML:

   ```bash
   zenml init
   ```

---

## Steps in the Pipeline

![Screenshot 2024-12-24 010040](https://github.com/user-attachments/assets/187ee207-9521-4316-8722-44592ccfe5b4)


### 1. Data Ingestion

```python
@step
def data_ingestion_step(file_path: str) -> pd.DataFrame:
    # Loads data using appropriate ingestor based on file extension.
    return DataIngestorFactory.get_data_ingestor(".zip").ingest(file_path)
```

- Input: Raw data file (ZIP format)
- Output: Pandas DataFrame

### 2. Missing Value Handling

```python
@step
def handle_missing_values_step(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    # Handles missing values using specified strategies.
    return MissingValueHandler(FillMissingValuesStrategy(method=strategy)).handle_missing_values(df)
```

- Options: `mean`, `median`, `mode`, `drop`

### 3. Feature Engineering

```python
@step
def feature_engineering_step(df: pd.DataFrame, strategy: str, features: list) -> pd.DataFrame:
    # Applies transformations like log scaling or standard scaling.
    return FeatureEngineer(LogTransformation(features)).apply_feature_engineering(df)
```

### 4. Outlier Detection

```python
@step
def outlier_detection_step(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    # Removes outliers based on Z-score.
    return OutlierDetector(ZScoreOutlierDetection(threshold=3)).handle_outliers(df, method="remove")
```

### 5. Model Training

```python
@step
def model_building_step(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    # Builds a scikit-learn pipeline with preprocessing and model training.
    return pipeline.fit(X_train, y_train)
```

### 6. Model Evaluation

```python
@step
def model_evaluator_step(trained_model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series):
    # Evaluates model performance.
    return ModelEvaluator(RegressionModelEvaluationStrategy()).evaluate(trained_model, X_test, y_test)
```

---

## Deployment

### Continuous Deployment Pipeline

The pipeline uses ZenML’s MLFlow integration for deployment:

```python
@pipeline
def continuous_deployment_pipeline():
    trained_model = ml_pipeline()
    mlflow_model_deployer_step(deploy_decision=True, model=trained_model)
```

### Batch Inference Pipeline

Run batch inference jobs with dynamically imported data:

```python
@pipeline
def inference_pipeline():
    batch_data = dynamic_importer()
    service = prediction_service_loader(pipeline_name="continuous_deployment_pipeline", step_name="mlflow_model_deployer_step")
    predictor(service=service, input_data=batch_data)
```

---


## License

This project is licensed under the MIT License 2.0. See the [LICENSE](LICENSE) file for details.

