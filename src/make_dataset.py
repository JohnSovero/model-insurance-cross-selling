# Script de Preparación de Datos
###################################

import pandas as pd
import numpy as np
import os


# Leemos los archivos csv
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/raw/', filename))
    print(filename, ' cargado correctamente')
    return df


# Realizamos la transformación de datos
def data_preparation(df):
    df["Gender"] = df["Gender"].replace({"Male": 0, "Female": 1}).astype("int32")
    df["Region_Code"] = df["Region_Code"].astype(int)
    df["Vehicle_Age"] = df["Vehicle_Age"].replace({"< 1 Year": 0, "1-2 Year": 1, "> 2 Years": 2}).astype("int32")
    df["Vehicle_Damage"] = df["Vehicle_Damage"].replace({"No": 0, "Yes": 1}).astype("int32")
    df["Annual_Premium"] = df["Annual_Premium"].astype(int)
    df["Policy_Sales_Channel"] = df["Policy_Sales_Channel"].astype(int)

    # Transformamos variables
    df["Previously_Insured_Annual_Premium"] = pd.factorize(
        df["Previously_Insured"].astype(str) + df["Annual_Premium"].astype(str)
    )[0]
    
    df["Previously_Insured_Vehicle_Age"] = pd.factorize(
        df["Previously_Insured"].astype(str) + df["Vehicle_Age"].astype(str)
    )[0]
    
    df["Previously_Insured_Vehicle_Damage"] = pd.factorize(
        df["Previously_Insured"].astype(str) + df["Vehicle_Damage"].astype(str)
    )[0]
    
    df["Previously_Insured_Vintage"] = pd.factorize(
        df["Previously_Insured"].astype(str) + df["Vintage"].astype(str)
    )[0]
    print('Transformación de datos completa')
    return df


# Exportamos la matriz de datos con las columnas seleccionadas
def data_exporting(df, features, filename):
    dfp = df[features]
    dfp.to_csv(os.path.join('../data/processed/', filename), index = False)
    print(filename, 'exportado correctamente en la carpeta processed')


# Generamos las matrices de datos que se necesitan para la implementación

def main():
    # Matriz de Entrenamiento
    df1 = read_file_csv('insurances.csv')
    tdf1 = data_preparation(df1)
    data_exporting(tdf1, ["Gender","Age","Previously_Insured","Vehicle_Age","Vehicle_Damage","Annual_Premium","Policy_Sales_Channel","Vintage","Previously_Insured_Annual_Premium","Previously_Insured_Vehicle_Damage","Previously_Insured_Vintage", "Response"],'insurances_train.csv')
    # Matriz de Validación
    df2 = read_file_csv('insurances_new.csv')
    tdf2 = data_preparation(df2)
    data_exporting(tdf2, ["Gender","Age","Previously_Insured","Vehicle_Age","Vehicle_Damage","Annual_Premium","Policy_Sales_Channel","Vintage","Previously_Insured_Annual_Premium","Previously_Insured_Vehicle_Damage","Previously_Insured_Vintage", "Response"],'insurances_val.csv')
    # Matriz de Scoring
    df3 = read_file_csv('insurances_score.csv')
    tdf3 = data_preparation(df3)
    data_exporting(tdf3, ["Gender","Age","Previously_Insured","Vehicle_Age","Vehicle_Damage","Annual_Premium","Policy_Sales_Channel","Vintage","Previously_Insured_Annual_Premium","Previously_Insured_Vehicle_Damage","Previously_Insured_Vintage"],'insurances_score.csv')
    
if __name__ == "__main__":
    main()