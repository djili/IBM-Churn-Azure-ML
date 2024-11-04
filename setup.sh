#! /usr/bin/sh

# Add name for custom name component

suffix="Abdou Khadre DIOP"

# Set the necessary variables
RESOURCE_GROUP="group-churn-telco-${suffix}"
RESOURCE_PROVIDER="Microsoft.MachineLearning"
RANDOM_REGION="eastus" # Choose between regions "eastus" "westus" "centralus" "northeurope" "westeurope"
WORKSPACE_NAME="workspace-churn-telco-${suffix}"
COMPUTE_INSTANCE="compute-instance-${suffix}"
COMPUTE_CLUSTER="aml-cluster-${suffix}"

# Setup Azure ML Provider
echo "Register the Machine Learning resource provider:"
az provider register --namespace $RESOURCE_PROVIDER

# Create the resource group and workspace
echo "Create a resource group and set as default:"
az group create --name $RESOURCE_GROUP --location $RANDOM_REGION
az configure --defaults group=$RESOURCE_GROUP

echo "Create an Azure Machine Learning workspace:"
az ml workspace create --name $WORKSPACE_NAME 
az configure --defaults workspace=$WORKSPACE_NAME 

# Create compute instance
echo "Creating a compute instance with name: " $COMPUTE_INSTANCE
az ml compute create --name ${COMPUTE_INSTANCE} --size STANDARD_DS11_V2 --type ComputeInstance 

# Create compute cluster
echo "Creating a compute cluster with name: " $COMPUTE_CLUSTER
az ml compute create --name ${COMPUTE_CLUSTER} --size STANDARD_DS11_V2 --max-instances 2 --type AmlCompute 

# Create data assets
echo "Create training data asset:"
az ml data create --type mltable --name "churn-data" --path ./data/Telco-Customer-Churn