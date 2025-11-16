# Código de Entrenamiento - Modelo de Predicción de Venta de Seguros a Clientes
############################################################################

import pandas as pd
import xgboost as xgb
import pickle
import os


# Cargar la tabla transformada
def score_model(filename, scores):
    df = pd.read_csv(os.path.join('../data/processed', filename))
    print(filename, ' cargado correctamente')
    # Leemos el modelo entrenado para usarlo
    package = '../models/best_model.pkl'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente')
    
    # Predecimos sobre el set de datos de Scoring    
    y_prob = model.predict_proba(df)[:, 1]
    threshold = 0.248621
    res = (y_prob >= threshold).astype(int)
    
    pred = pd.DataFrame(res, columns=['Response'])
    pred.to_csv(os.path.join('../data/scores/', scores), index = False)
    print(scores, 'exportado correctamente en la carpeta scores')


# Scoring desde el inicio
def main():
    df = score_model('insurances_score.csv','final_score.csv')
    print('Finalizó el Scoring del Modelo')


if __name__ == "__main__":
    main()