from fastapi import FastAPI, Body
# from typing import Optional, List
import pandas as pd
from joblib import load
from pydantic import BaseModel
import logging
import json
import pickle
import base64

app = FastAPI()

# Configuration du gestionnaire de logs pour écrire dans un fichier
log_file = "app.log"
handler = logging.FileHandler(log_file)
handler.setLevel(logging.DEBUG)

# Formatter pour les messages de logs
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Ajouter le gestionnaire de logs au logger racine
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(handler)

@app.get("/")
async def root():
    return {"message": "API pour Dashboad"}

@app.on_event('startup')                    # charger le modèle et les données au démarrage de l'application
def load_model():       
    global clf_model, shap_model, data_df, shap_values_all_sample
    # clf_model = load('LGB_shap.joblib')   # modèle lightgbm
    clf_model = load('XGB2s.joblib')        # modèle xgboost
    shap_model = load('explainer.joblib')   # modèle calculant la shap value
    data_df = load('df_val.joblib')         # contient les données de validation + prédition des prabas réalisées en offline      
    shap_values_all_sample = load('shap_values_all_sample.joblib')    # contient toutes les shap values calculées en off-line
    

# class ClientData(BaseModel):
    # data: list[float]

 # ...

@app.post('/predict', tags=["predictions"])
async def get_prediction(client_id: int):
   
    # Filtrage du dataframe : retenir la ligne du client et les colonnes des features
    features = data_df.columns.drop(["SK_ID_CURR", "target", "y_probs"])
    input_data = data_df[data_df["SK_ID_CURR"]==int(client_id)]
    input_data = input_data[features]
    
    if len(input_data) == 0:        # Si le dataframe est vide de lignes
        return {"error": "Client ID not found"}
    
    # Prédition selon le modèle
    prediction = clf_model.predict(input_data)[0]
    prediction = int(prediction)
    
    # Préparation de la réponse (la prédiction)
    response_data = {
        "prediction": prediction,
        "client_id": client_id
    }
    
    # Ecriture dans le fichier log et renvoie du résultat
    root_logger.info(f"Prediction for client ID {client_id}: {prediction}")
    return response_data

@app.post('/predict_prob', tags=["predictions_prob"])
async def get_prediction_prob(client_id: int):
    
    # Filtrage du dataframe : retenir la ligne du client et les colonnes des features
    features = data_df.columns.drop(["SK_ID_CURR", "target", "y_probs"])
    input_data = data_df[data_df["SK_ID_CURR"]==int(client_id)]
    input_data = input_data[features]
    
    if len(input_data) == 0:            # Si le dataframe est vide de lignes
        return {"error": "Client ID not found"}
    
    # Prédition selon le modèle
    prediction = clf_model.predict_proba(input_data)[:, 1]
    prediction = float(prediction)
    prediction = round(prediction,6)
    
    # Préparation de la réponse (la prédiction)
    response_data = {
        "prediction_prob": prediction,
        "client_id": client_id
    }
    
    # Ecriture dans le fichier log et renvoie du résultat
    root_logger.info(f"Prediction (probability) for client ID {client_id}: {prediction}")
    return response_data

@app.post('/similar_cl', tags=["similar_clients"])
async def similar_client(client_id: int):

    # ========= Code permettant le calcul totalement en on-line des clients similaires ===================================

    # features = data_df.columns.drop(["SK_ID_CURR", "target"])

    # data_client = data_df[data_df["SK_ID_CURR"]==int(client_id)]      # Calcul de la probabilité du client
    # proba = clf_model.predict_proba(data_client[features])[:, 1]
    # proba = float(proba)
    # proba = round(proba,6)

    # df_result = data_df[features]                                     # Calcul des probabilités de tous les clients
    # probas = clf_model.predict_proba(df_result)
    # df_result['y_probs'] = probas[:,1]
     
    # df_result['diff_probas'] = abs(df_result["y_probs"] - proba)      # Tri des clients selon leur proximité par rapport
    # df = df_result.sort_values(by='diff_probas', ascending=True)      # au client en question en terme de probabilité

    # ========= Code utilisant la méthode déjà définie et le calcul en off-line des probas des autres clients =============

    proba_json = await get_prediction_prob(client_id)
    proba = float(proba_json["prediction_prob"])   
    
    df = data_df.drop(['target'], axis=1)
    df ['diff_probas'] = abs(df["y_probs"] - proba)    
    df = df.sort_values(by='diff_probas', ascending=True)
    
    # ========== Code pour retenir et renvoyer les 5 plus proches clients ==================================================
    
    nosc = df.value_counts().sum()      # nosc : number_of_similat_clients
    if nosc < 5:
        df = df.head(nosc) 
    else:    
        df = df.head(5)
    return df.to_dict(orient='records')

    
@app.post('/shap_val', tags=["shap_value"])
async def get_shap_value (client_id: int):
    # Filtrage du dataframe : retenir la ligne du client et les colonnes des features
    features = data_df.columns
    features = features.drop(["SK_ID_CURR", "target", "y_probs"])
    input_data = data_df[data_df["SK_ID_CURR"]==int(client_id)]
    input_data = input_data[features]

    # Calcul des SHAP values selon le modèle (correspondant au client en question)
    shap_values = shap_model(input_data)
    
    # Serialisation, encodage et renvoie du résultat (json)
    shap_values_bytes = pickle.dumps(shap_values)
    shap_values_encoded = base64.b64encode(shap_values_bytes).decode("utf-8")
    return {"shape_values": shap_values_encoded}

@app.post('/shap_val_all', tags=["shap_values_all"])
async def get_shap_value_all ():
    # Toutes les SHAP values (calculées et échantiollonnées en off-line) sont chargée au démarrage de l'application 
    # Ici, ces valeurs sont sérialisées et encodées pour être renvoyées (en json)
    shap_values_all_sample_bytes = pickle.dumps(shap_values_all_sample)
    shap_values_all_sample_encoded = base64.b64encode(shap_values_all_sample_bytes).decode("utf-8")
    return {"shape_values_all": shap_values_all_sample_encoded}
# ...

# if __name__ == "__main__":
    # import uvicorn
    # uvicorn.run(app, host="127.0.0.1", port=5000)
