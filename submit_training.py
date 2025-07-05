from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import os
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

# Récupérer les informations d'identification
subscription_id = os.getenv("SUBSCRIPTION_ID")
resource_group = os.getenv("RESOURCE_GROUP")
workspace_name = os.getenv("WORKSPACE_NAME")
suffix = os.getenv("SUFFIX")

# Se connecter à l'espace de travail Azure ML
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name,
)

# Créer le job d'entraînement
from azure.ai.ml import load_job

# Charger la configuration du job
train_job = load_job("train_job.yml")

# Remplacer la variable ${suffix} dans le nom de la compute instance
train_job.compute = train_job.compute.replace("${suffix}", suffix.lower())

# Soumettre le job
print("Soumission du job d'entraînement...")
returned_job = ml_client.jobs.create_or_update(train_job)

# Afficher le lien vers le portail Azure ML
print(f"Job soumis avec l'ID: {returned_job.name}")
print(f"Suivez la progression sur: {returned_job.studio_url}")
