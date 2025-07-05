import os
import json
import pandas as pd
import numpy as np
from azureml.core.model import Model
import joblib

def init():
    """
    Initialise le modèle
    """
    global model
    
    # Obtenir le chemin du modèle
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model')
    
    # Charger le modèle
    model = joblib.load(os.path.join(model_path, 'model.pkl'))

def run(raw_data):
    """
    Effectue une prédiction à partir des données d'entrée
    
    Args:
        raw_data (str): Données d'entrée au format JSON
        
    Returns:
        list: Prédictions du modèle
    """
    try:
        # Convertir les données JSON en DataFrame
        data = json.loads(raw_data)['data']
        input_df = pd.DataFrame(data)
        
        # Effectuer la prédiction
        predictions = model.predict_proba(input_df)
        
        # Retourner les probabilités de churn (seulement la classe positive)
        return json.dumps({"predictions": predictions[:, 1].tolist()})
        
    except Exception as e:
        error = str(e)
        return json.dumps({"error": error})
