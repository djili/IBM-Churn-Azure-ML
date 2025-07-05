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



