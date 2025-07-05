from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import argparse
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from azure.ai.ml.sweep import Normal, Uniform

def main(args):
    # lecture des données
    df = getData(args.training_data)
    # Split des données
    X_train, X_test, y_train, y_test = splitData(df)
    # Entrainer le modèle
    model = trainModel(args.reg_rate, X_train, X_test, y_train, y_test)
    # Évaluer le modèle
    evalModel(model, X_test, y_test)
    # Enregistrer le modèle
    registerModel(model, args.model_name)




def getData(data_asset_Name):
    # Charger les variables d'environnement
    from dotenv import load_dotenv
    import os
    
    # Charger les variables depuis .env
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    load_dotenv(env_path)
    
    # Authentification
    try:
        credential = DefaultAzureCredential()
    except Exception as ex:
        credential = InteractiveBrowserCredential()
    
    # Récupérer les informations de l'espace de travail depuis les variables d'environnement
    subscription_id = os.getenv("SUBSCRIPTION_ID")
    resource_group = os.getenv("RESOURCE_GROUP")
    workspace_name = os.getenv("WORKSPACE_NAME")
    
    if not all([subscription_id, resource_group, workspace_name]):
        raise ValueError("Veuillez configurer les variables d'environnement SUBSCRIPTION_ID, RESOURCE_GROUP et WORKSPACE_NAME dans le fichier .env")
    
    # Initialiser le client ML
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )
    # datapath
    data_asset = ml_client.data.get(data_asset_Name, version="lastest")
    # datafile
    data = pd.read_csv(data_asset.path)
    return data

# Fonction pour separer les donnees
def splitData(df):
    # Sélection des colonnes pertinentes pour l'entraînement
    selected_columns = ['Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Tenure Months', 'Phone Service',
                        'Multiple Lines', 'Internet Service', 'Online Security', 'Online Backup', 'Device Protection',
                        'Tech Support', 'Streaming TV', 'Streaming Movies', 'Contract', 'Paperless Billing', 'Payment Method',
                        'Monthly Charges', 'Total Charges']
    # Filtrer les données avec les colonnes sélectionnées
    data_selected = df[selected_columns + ['Churn Value']]
    # Conversion des colonnes catégorielles en numériques
    data_encoded = pd.get_dummies(data_selected, drop_first=True)
    # Séparer les caractéristiques (features) et la cible (target)
    X = data_encoded.drop('Churn Value', axis=1)
    y = data_encoded['Churn Value']
    # Séparation des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Fonction pour entraîner le modèle et finetuning des hyperparametres 
def trainModel(reg_rate, X_train, X_test, y_train, y_test):
    mlflow.log_param("Regularization rate", reg_rate)

    commandForJobSweep = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': [0.01, 0.1, 1],
    'degree': [2, 3, 4]  # seulement pour 'poly'
    }
    print("Training model...")
    model =SVC(probability=True, random_state=42).fit(X_train, y_train)
    return model

# Fonction pour evaluer le modèle
def evalModel(model, X_test, y_test):
    # Calcul de accuracy
    y_hat = model.predict(X_test)
    acc = np.average(y_hat == y_test)
    print('Accuracy:', acc)
    mlflow.log_metric("Accuracy", acc)

    # Calcul AUC
    y_scores = model.predict_proba(X_test)
    auc = roc_auc_score(y_test,y_scores[:,1])
    print('AUC: ' + str(auc))
    mlflow.log_metric("AUC", auc)

    # Tracer courbe ROC 
    fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
    fig = plt.figure(figsize=(6, 4))
    # Tracer de la diagonale
    plt.plot([0, 1], [0, 1], 'k--')
    # Tracage du taux de faux positifs et du taux de vrais positifs
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig("ROC-Curve.png")
    mlflow.log_artifact("ROC-Curve.png")  

def parse_args():
    #configurer le parser 
    parser = argparse.ArgumentParser()
    # Ajout des arguments
    parser.add_argument("--experiment_name", type=str, default="churn_prediction",
                      help="Nom de l'expérience Azure ML")
    parser.add_argument("--model_name", type=str, default="ChurnPredictionModel",
                      help="Nom du modèle dans le registre Azure ML")
    parser.add_argument("--training_data", dest='training_data',type=str)
    parser.add_argument("--reg_rate", dest='reg_rate',type=float, default=0.01)
    # Parsage des arguments
    args = parser.parse_args()
    # return args
    return args

def registerModel(model, model_name):
    """
    Enregistre le modèle dans Azure ML et le sauvegarde localement
    
    Args:
        model: Modèle entraîné à enregistrer
        model_name (str): Nom à donner au modèle dans le registre Azure ML
    """
    print(f"Enregistrement du modèle {model_name}...")
    
    # Enregistrement dans le registre Azure ML
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=model_name
    )
    
    # Sauvegarde locale pour référence
    import os
    os.makedirs("outputs", exist_ok=True)
    model_path = os.path.join("outputs", f"{model_name.lower().replace(' ', '_')}")
    mlflow.sklearn.save_model(
        model,
        path=model_path,
    )
    print(f"Modèle enregistré avec succès dans {model_path}")

# run script
if __name__ == "__main__":
    # Ajouter un espace dans les logs 
    print("\n\n")
    
    # Démarrer le suivi MLflow
    mlflow.start_run()
    print("*" * 60)
    # Parsage des arguments 
    args = parse_args()
    
    # Configurer le nom de l'expérience
    mlflow.set_experiment(args.experiment_name)
    
    try:
        main(args)
        mlflow.end_run(status="FINISHED")
    except Exception as e:
        print(f"Erreur lors de l'exécution: {str(e)}")
        mlflow.end_run(status="FAILED")
    # Ajouter un espace dans les logs 
    print("*" * 60)
    print("\n\n")
