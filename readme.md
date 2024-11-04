# IBM Telco Customer Churn Prediction Project With Azure ML

## Description

This project aims to predict customer churn using the IBM Telco Customer Churn dataset. By leveraging Azure Machine Learning's capabilities, we create a scalable and explainable machine learning solution that can be used to anticipate churn behavior, enabling proactive customer retention strategies. The project utilizes several advanced Azure ML features, including automated machine learning pipelines, version tracking, model deployment, and responsible AI tools.

## Dataset

The dataset is publicly available and can be downloaded from IBM's website:

- **IBM Telco Customer Churn Dataset**: [Download here](https://www.ibm.com/communities/analytics/watson-analytics-blog/guide-to-sample-datasets/](https://www.ibm.com/docs/en/cognos-analytics/11.1.0?topic=samples-telco-customer-churn))

The dataset contains information about customer demographics, account details, services subscribed to, and whether or not the customer churned. This data will be used to develop a classification model to predict the likelihood of churn.

## Project Structure and Components

### 1. Data Analysis

- **Exploratory Data Analysis (EDA)**: Conduct a thorough EDA to understand the dataset's structure, including distributions, missing values, and correlations.
- **Visualization**: Use `matplotlib` and `seaborn` for visualizations to uncover insights and relationships, such as churn rates across different demographics, tenure, and service usage.
- **Data Cleaning**: Handle missing values, correct anomalies, and prepare the data for feature engineering.

### 2. Environment Setup on Azure ML

- Set up an Azure ML Workspace, which will serve as the centralized platform for managing resources, tracking experiments, and deploying models.
- Define and configure a virtual environment for the project with necessary dependencies, including `scikit-learn`, `pandas`, `azureml-sdk`, `matplotlib`, `mlflow`, and `seaborn`.

### 3. Experiment Tracking with MLflow

- Use MLflow integrated within Azure ML to track model performance, parameters, metrics, and other experiment details for each run.
- Track and compare different model versions to identify the best-performing model for churn prediction.

### 4. Data Processing and Feature Engineering

- Clean the dataset (handle missing values, categorical encoding, scaling numerical features).
- Engineer features that capture customer engagement, tenure, and service usage.
- Store the processed data for reproducibility in the Azure ML datastore.

### 5. Model Training and Hyperparameter Tuning

- Train multiple classification models, including logistic regression, random forests, and gradient-boosted trees, using Azure ML’s automated ML pipelines.
- Tune hyperparameters to optimize the model for predictive performance.
- Log experiment runs and selected metrics to MLflow.

### 6. Model Evaluation and Interpretability

- Evaluate models based on metrics like accuracy, AUC, precision, and recall to measure predictive performance.
- Use SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-Agnostic Explanations) for interpretability, providing transparency on factors driving the churn predictions.

### 7. Model Deployment with Real-time Endpoints

- Deploy the best model to an Azure Container Instance or Kubernetes cluster.
- Expose the model through a RESTful API endpoint, allowing for real-time predictions.
- Set up monitoring on the endpoint to track latency, errors, and usage for ongoing performance analysis.

### 8. Responsible AI Dashboard

- Utilize Azure ML's Responsible AI tools to examine model fairness, transparency, and reliability.
- Configure the Responsible AI dashboard to:
  - Identify potential biases in the model's predictions.
  - Ensure feature contributions are equitable across different customer segments.
  - Set up alerts for model drift or significant deviations in model performance over time.

## Azure ML Workflow Overview

1. **Preprocessing**: Data ingestion, cleaning, and feature engineering.
2. **Experiment Tracking**: Experiment logging and comparison with MLflow.
3. **Model Training and Tuning**: Automated ML pipelines for efficient model tuning.
4. **Evaluation and Interpretability**: Metrics logging and use of explainable AI methods.
5. **Deployment**: Model deployment and monitoring through Azure endpoints.
6. **Responsible AI**: Bias detection, transparency, and continuous monitoring with the Responsible AI dashboard.

## Requirements

- Python 3.7+
- Azure ML SDK
- MLflow
- Pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn

To install the necessary libraries, run:

```bash
pip install -r requirements.txt
```

## Author

Made with ❤️

Name : Abdou Khadre DIOP
Email : diopous1@gmail.com



