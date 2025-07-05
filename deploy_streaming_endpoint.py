from azure.ai.ml import MLClient, Model
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Environment,
    CodeConfiguration,
    ModelConfiguration,
)
from azure.identity import DefaultAzureCredential
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Définir le chemin racine du projet
ROOT_DIR = Path(__file__).parent.absolute()

# Charger les variables d'environnement
load_dotenv(ROOT_DIR / ".env")

# Ajouter le répertoire src au chemin Python
sys.path.append(str(ROOT_DIR / "src"))

# Récupérer les informations d'identification
subscription_id = os.getenv("SUBSCRIPTION_ID")
resource_group = os.getenv("RESOURCE_GROUP")
workspace_name = os.getenv("WORKSPACE_NAME")
suffix = os.getenv("SUFFIX")

# Paramètres de déploiement
ENDPOINT_NAME = f"churn-streaming-endpoint-{suffix}"
DEPLOYMENT_NAME = "blue"
MODEL_NAME = "ChurnPredictionModel"
COMPUTE_CLUSTER = f"aml-cluster-{suffix}"

# Se connecter à l'espace de travail Azure ML
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name,
)

def deploy_streaming_endpoint():
    """
    Déploie un point de terminaison de streaming pour le modèle de prédiction de churn
    """
    print("Création du point de terminaison de streaming...")
    
    # 1. Créer ou mettre à jour le point de terminaison
    endpoint = ManagedOnlineEndpoint(
        name=ENDPOINT_NAME,
        description="Point de terminaison pour la prédiction de churn en temps réel",
        auth_mode="key",
    )

    # Créer le point de terminaison s'il n'existe pas
    try:
        ml_client.online_endpoints.get(name=ENDPOINT_NAME)
        print(f"Le point de terminaison {ENDPOINT_NAME} existe déjà.")
    except Exception:
        print(f"Création du point de terminaison {ENDPOINT_NAME}...")
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    # 2. Récupérer la dernière version du modèle
    print(f"Récupération du modèle {MODEL_NAME}...")
    model = ml_client.models.get(name=MODEL_NAME, label="latest")

    # 3. Créer l'environnement de déploiement
    env = Environment(
        name="churn-env",
        description="Environnement pour le modèle de prédiction de churn",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
        conda_file="conda-env.yml",
    )
    
    # Vérifier que le fichier conda existe
    if not os.path.exists("conda-env.yml"):
        raise FileNotFoundError("Le fichier conda-env.yml est introuvable dans le répertoire courant.")

    # 4. Créer la configuration de déploiement
    deployment = ManagedOnlineDeployment(
        name=DEPLOYMENT_NAME,
        endpoint_name=ENDPOINT_NAME,
        model=model,
        environment=env,
        code_configuration=CodeConfiguration(
            code="./src",
            scoring_script="score.py",
        ),
        instance_type="Standard_DS3_v2",
        instance_count=1,
    )

    # 5. Déployer le modèle
    print(f"Déploiement du modèle sur le point de terminaison {ENDPOINT_NAME}...")
    ml_client.online_deployments.begin_create_or_update(deployment).result()

    # 6. Mettre à jour le trafic (100% vers le nouveau déploiement)
    endpoint.traffic = {DEPLOYMENT_NAME: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    # Afficher les informations du point de terminaison
    endpoint = ml_client.online_endpoints.get(name=ENDPOINT_NAME)
    print(f"\nPoint de terminaison déployé avec succès !")
    print(f"Nom du point de terminaison: {endpoint.name}")
    print(f"URI du point de terminaison: {endpoint.scoring_uri}")
    print(f"Clé d'authentification: {ml_client.online_endpoints.get_keys(name=ENDPOINT_NAME).primary_key}")

if __name__ == "__main__":
    deploy_streaming_endpoint()
