from fastapi import FastAPI
import uvicorn
import pandas as pd
import json as json


dfMoviesFinal = pd.read_parquet('DataSets/dfMoviesFinal.parquet')
dfCreditsFinal = pd.read_parquet('DataSets/dfCreditsFinal.parquet')

def cantidad_filmaciones_mes(mes: str):
    # Diccionario para traducir los meses en español a sus números correspondientes
    meses_dict = {
        'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
        'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
    }
    mes = mes.lower()
    if mes not in meses_dict:
        raise ValueError("Mes no válido. Debe ser uno de los siguientes: enero, febrero, marzo, abril, mayo, junio, julio, agosto, septiembre, octubre, noviembre, diciembre.")
    mes_numero = meses_dict[mes]
 
    dfMoviesFinal['mes'] = dfMoviesFinal['release_date'].dt.month
    df_mes = dfMoviesFinal[dfMoviesFinal['mes'] == mes_numero]
    cantidad = df_mes['id_pelicula'].nunique()
    
    return f"{cantidad} películas fueron estrenadas en el mes de {mes}"


print(cantidad_filmaciones_mes('diciembre'))
