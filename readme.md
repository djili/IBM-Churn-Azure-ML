# IBM Telco Customer Churn Prediction Project With Azure ML

## Description

This project aims to predict customer churn using the IBM Telco Customer Churn dataset. By leveraging Azure Machine Learning's capabilities, we create a scalable and explainable machine learning solution that can be used to anticipate churn behavior, enabling proactive customer retention strategies. The project utilizes several advanced Azure ML features, including:

- Automated machine learning pipelines
- Model versioning and tracking with MLflow
- Real-time model deployment with streaming endpoints
- Model monitoring and management
- Responsible AI tools for model interpretability

## Dataset

The dataset is publicly available and can be downloaded from IBM's website:

- **IBM Telco Customer Churn Dataset**: [Download here](https://www.ibm.com/communities/analytics/watson-analytics-blog/guide-to-sample-datasets/](https://www.ibm.com/docs/en/cognos-analytics/11.1.0?topic=samples-telco-customer-churn))

The dataset contains information about customer demographics, account details, services subscribed to, and whether or not the customer churned. This data will be used to develop a classification model to predict the likelihood of churn.

## Project Structure and Components

### 1. Data Exploration and Analysis

#### Exploratory Data Analysis (EDA)

Before model training, we conducted a comprehensive EDA to understand the dataset's characteristics:

```python
# Sample EDA code
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load and preview data
df = pd.read_csv('data/telco_churn.csv')
print(df.info())
print(df.describe())

# Visualize churn distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Churn', data=df)
plt.title('Distribution of Churn')
plt.show()

# Analyze correlation between numerical features
plt.figure(figsize=(12, 8))
sns.heatmap(df.select_dtypes(include=['float64', 'int64']).corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
```

Key findings from EDA:
- **Class Imbalance**: The dataset shows an imbalance in churn classes (approximately 73% non-churn vs 27% churn)
- **Important Features**: Tenure, Monthly Charges, and Total Charges show significant correlation with churn
- **Missing Values**: Handled by imputation or removal based on feature importance
- **Categorical Variables**: Encoded using one-hot encoding for model compatibility

#### Data Preprocessing

- **Handling Missing Values**:
  - Numerical features: Mean imputation
  - Categorical features: Mode imputation
  
- **Feature Engineering**:
  - Created interaction terms between important features
  - Binned continuous variables where appropriate
  - Encoded categorical variables using one-hot encoding
  
- **Train-Test Split**:
  - 80% training, 20% testing
  - Stratified split to maintain class distribution
  - Random state fixed for reproducibility

### 2. Environment Setup on Azure ML

#### Azure ML Workspace Setup

1. **Create Azure ML Workspace**
   ```bash
   # Install Azure ML CLI extension
   az extension add -n azure-cli-ml
   
   # Create resource group
   az group create --name your-rg --location eastus
   
   # Create Azure ML workspace
   az ml workspace create -w your-workspace -g your-rg
   ```

2. **Configure Local Environment**
   - Install required packages:
     ```bash
     pip install azureml-sdk pandas scikit-learn mlflow matplotlib seaborn xgboost
     ```
   - Set up Azure ML environment configuration
   - Configure authentication (Service Principal or Interactive)

#### Development Environment

We recommend using Jupyter Notebooks or VS Code with the following extensions:
- Python extension
- Azure ML extension
- Jupyter extension

Example notebook for environment setup:

```python
from azureml.core import Workspace, Experiment, Environment

# Load workspace
ws = Workspace.from_config()

# Create experiment
experiment = Experiment(workspace=ws, name='churn-prediction')

# Define environment
env = Environment.from_conda_specification(
    name='churn-env',
    file_path='conda-env.yml'
)
```

### 3. Experiment Tracking with MLflow

- Use MLflow integrated within Azure ML to track model performance, parameters, metrics, and other experiment details for each run.
- Track and compare different model versions to identify the best-performing model for churn prediction.

### 4. Data Processing and Feature Engineering

- Clean the dataset (handle missing values, categorical encoding, scaling numerical features).
- Engineer features that capture customer engagement, tenure, and service usage.
- Store the processed data for reproducibility in the Azure ML datastore.

### 5. Model Selection and Training

#### Model Selection

We evaluated several classification algorithms to identify the best performing model for our churn prediction task:

1. **Logistic Regression**
   - Baseline model for binary classification
   - Good interpretability but limited by linear decision boundaries

2. **Random Forest**
   - Handles non-linear relationships well
   - Robust to outliers and noise
   - Provides feature importance scores

3. **Gradient Boosting (XGBoost/LightGBM)**
   - Often provides best performance
   - Handles class imbalance well
   - Requires careful hyperparameter tuning

4. **Support Vector Machine (SVM)**
   - Effective in high-dimensional spaces
   - Memory intensive for large datasets

#### Model Training and Hyperparameter Tuning

We used Azure ML's hyperparameter tuning capabilities to optimize our models:

```python
from azureml.train.hyperdrive import RandomParameterSampling, BanditPolicy, HyperDriveConfig, PrimaryMetricGoal
from azureml.train.hyperdrive import choice, uniform

# Define parameter search space
param_sampling = RandomParameterSampling({
    '--learning_rate': uniform(0.01, 0.1),
    '--n_estimators': choice(50, 100, 150, 200),
    '--max_depth': choice(3, 5, 7, 10),
    '--min_samples_split': choice(2, 5, 10),
    '--min_samples_leaf': choice(1, 2, 4)
})

# Configure hyperdrive
hyperdrive_config = HyperDriveConfig(
    estimator=estimator,
    hyperparameter_sampling=param_sampling,
    policy=BanditPolicy(evaluation_interval=2, slack_factor=0.1),
    primary_metric_name='AUC',
    primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
    max_total_runs=20,
    max_concurrent_runs=4
)
```

#### Model Evaluation Metrics

We evaluated models using multiple metrics to ensure balanced performance:

- **AUC-ROC**: Primary metric for model selection (handles class imbalance well)
- **Precision and Recall**: Important for business impact (balancing false positives/negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visualizing true/false positives/negatives

#### Selected Model

After thorough evaluation, we selected **XGBoost** as our final model due to its superior performance in terms of AUC-ROC and F1-score. The model was then fine-tuned using cross-validation and deployed for inference.

Key hyperparameters of the final model:
- Learning rate: 0.1
- Max depth: 7
- N_estimators: 150
- Subsample: 0.8
- Colsample_bytree: 0.8

### 6. Model Evaluation and Interpretability

- Evaluate models based on metrics like accuracy, AUC, precision, and recall to measure predictive performance.
- Use SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-Agnostic Explanations) for interpretability, providing transparency on factors driving the churn predictions.

### 7. Model Training and Deployment

#### Training the Model

To train the model using Azure ML, use the following command:

```bash
python submit_training.py
```

This will:
1. Create an Azure ML experiment
2. Train the model using the specified compute instance
3. Track metrics and artifacts with MLflow
4. Register the trained model in Azure ML Model Registry

#### Real-time Deployment

Deploy the model as a real-time streaming endpoint:

```bash
python deploy_streaming_endpoint.py
```

This will:
1. Create a streaming endpoint in Azure ML
2. Deploy the latest registered model
3. Set up the scoring environment with the specified Conda environment
4. Return the endpoint URL and authentication key

#### Testing the Endpoint

You can test the deployed endpoint using the following Python code:

```python
import requests
import json

# Replace with your endpoint URL and key
ENDPOINT_URL = "YOUR_ENDPOINT_URL"
API_KEY = "YOUR_API_KEY"

# Sample input data
data = {
    "data": [{
        "Tenure Months": 12,
        "Monthly Charges": 70.5,
        "Total Charges": 850.0,
        # ... other features
    }]
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

response = requests.post(ENDPOINT_URL, json=data, headers=headers)
print("Prediction:", response.json())
```

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

## Environment Setup

### Prerequisites

- Python 3.8
- Azure CLI
- Azure ML extension for CLI

### Create Conda Environment

```bash
conda env create -f conda-env.yml
conda activate basic-env-cpu
```

### Install Additional Dependencies

```bash
pip install azure-ai-ml azure-identity python-dotenv
```

### Configure Azure Authentication

1. Log in to Azure:
   ```bash
   az login
   ```

2. Set your subscription:
   ```bash
   az account set --subscription <subscription-id>
   ```

3. Create a `.env` file with your Azure ML workspace details:
   ```env
   SUBSCRIPTION_ID=your-subscription-id
   RESOURCE_GROUP=your-resource-group
   WORKSPACE_NAME=your-workspace-name
   SUFFIX=unique-suffix
   ```

## Project Structure

```
.
├── .env                    # Environment variables
├── conda-env.yml           # Conda environment specification
├── deploy_streaming_endpoint.py  # Deployment script
├── submit_training.py      # Training submission script
├── train_job.yml           # Training job configuration
└── src/
    ├── train.py           # Training pipeline
    └── score.py           # Scoring script for deployment
```

## Monitoring and Maintenance

- Monitor your deployed endpoints in the Azure ML Studio
- Set up alerts for model drift and performance degradation
- Regularly update the model with new training data
- Use Azure Monitor for detailed logging and metrics

## Troubleshooting

- **Authentication Errors**: Ensure your Azure CLI is properly authenticated
- **Environment Issues**: Verify all dependencies are installed correctly using `conda list`
- **Deployment Failures**: Check the Azure ML Studio logs for detailed error messages
- **Scoring Errors**: Ensure the input data matches the expected schema

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Made with ❤️

Name: Abdou Khadre DIOP  
Email: diopous1@gmail.com  
GitHub: [@djili](https://github.com/djili)



