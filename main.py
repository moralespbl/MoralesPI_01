from fastapi import FastAPI
import uvicorn
import pandas as pd
import json as json

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk


app = FastAPI()

# uvicorn main:app --reload 

# Cargar data frames
dfMoviesFinal = pd.read_parquet('DataSets/dfMoviesSintetico.parquet')
dfCrewFinal = pd.read_parquet('DataSets/dfCrew.parquet')
dfCastFinal = pd.read_parquet('DataSets/dfCastFinal.parquet')


# Convertir la columna 'id_pelicula' a tipo float DataFrames
dfMoviesFinal['id_pelicula'] = dfMoviesFinal['id_pelicula'].astype(float)
dfCrewFinal['id_pelicula'] = dfCrewFinal['id_pelicula'].astype(float)
dfCastFinal['id_pelicula'] = dfCastFinal['id_pelicula'].astype(float)


# Convertir la columna 'release_date' a tipo datetime
dfMoviesFinal['release_date'] = pd.to_datetime(dfMoviesFinal['release_date'])

@app.get("/filmaciones_por_mes/{mes}")
def cantidad_filmaciones_mes(mes: str):
    meses_dict = {
        'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
        'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
    }
    mes = mes.lower()
    if mes not in meses_dict:
        raise ValueError("Error ingrese uno de los meses del año en letras por favor.")
    mes_numero = meses_dict[mes]

    cantidad = dfMoviesFinal[dfMoviesFinal['release_date'].dt.month == mes_numero]['id_pelicula'].nunique()
    
    return f"{cantidad} películas fueron estrenadas en el mes de {mes}"


@app.get("/filmaciones/dia/{dia}")
def cantidad_filmaciones_dia(dia):

    dias_dict = {
        'lunes': 1, 'martes': 2, 'miercoles': 3, 'jueves': 4, 'viernes': 5, 'sabado': 6, 'domingo': 7
    }
    dia = dia.lower()

    if dia not in dias_dict:
        raise ValueError("Día no válido. Debe ser uno de los siguientes: lunes, martes, miércoles, jueves, viernes, sábado, domingo.")
    
    dia_numero = dias_dict[dia]

    dfMoviesFinal['dia_semana'] = dfMoviesFinal['release_date'].dt.dayofweek + 1

    cantidad = dfMoviesFinal[dfMoviesFinal['dia_semana'] == dia_numero]['id_pelicula'].nunique()
    
    return f"{cantidad} cantidad de películas fueron estrenadas en los días {dia}"


@app.get("/peliculas/score/{titulo}")
def score_titulo(nombre_peli: str):

    pelicula = dfMoviesFinal[dfMoviesFinal['title'].str.contains(nombre_peli, case=False, na=False)]

    if pelicula.empty:
        return f"No se encontró la película '{nombre_peli}'."

    titulo = pelicula.iloc[0]['title']
    anio_estreno = pelicula.iloc[0]['release_year']
    puntuacion = pelicula.iloc[0]['vote_average']

    return f"La película '{titulo}' fue estrenada en el año {anio_estreno} con un score/popularidad de {puntuacion}."


@app.get("/peliculas/votos/{titulo}")
def votos_titulo(titulo_de_la_filmacion: str):
    pelicula = dfMoviesFinal[dfMoviesFinal['title'].str.contains(titulo_de_la_filmacion, case=False, na=False)]
    
    if pelicula.empty:
        return f"No se encontró la película '{titulo_de_la_filmacion}'."
    
    titulo = pelicula.iloc[0]['title']
    anio_estreno = pelicula.iloc[0]['release_year']
    cantidad_votos = pelicula.iloc[0]['vote_count']
    promedio_votos = pelicula.iloc[0]['vote_average']
    
    if cantidad_votos < 2000:
        return f"La película '{titulo}' no cumple con la condición de tener al menos 2000 valoraciones (tiene {cantidad_votos} valoraciones)."
    
    return (f"La película '{titulo}' fue estrenada en el año {anio_estreno}. "
            f"La misma cuenta con un total de {cantidad_votos} valoraciones, con un promedio de {promedio_votos}.")


@app.get("/actor/{nombre_actor}")
def get_actor(nombre_actor: str):
    # Filtrar el DataFrame dfCreditsFinal para encontrar las películas en las que ha actuado el actor
    peliculas_actor = dfCastFinal[dfCastFinal['name_cast'].str.contains(nombre_actor, case=False, na=False)]

    # Verificar si el actor aparece en el dataset
    if peliculas_actor.empty:
        return f"No se encontró al actor {nombre_actor} en el dataset."

    # Obtener las películas únicas en las que ha participado, eliminando duplicados de 'id_pelicula'
    ids_peliculas = peliculas_actor['id_pelicula'].drop_duplicates()

    # Filtrar las películas en el DataFrame dfMoviesFinal usando los IDs obtenidos
    peliculas_info = dfMoviesFinal[dfMoviesFinal['id_pelicula'].isin(ids_peliculas)]
    peliculas_info = peliculas_info.drop_duplicates(subset='id_pelicula')

    # Calcular el retorno total y promedio
    retorno_total = peliculas_info['return'].sum()
    retorno_promedio = retorno_total/len(ids_peliculas)

    # Devolver la cantidad de películas y el retorno total y promedio
    cantidad_peliculas = len(ids_peliculas)
    if cantidad_peliculas == 0:
        return f"El actor {nombre_actor} no tiene películas con presupuesto válido."

    return (f"El actor {nombre_actor} ha participado en {cantidad_peliculas} películas, "
            f"con un retorno total de {retorno_total*100:.2f}% y un promedio de {retorno_promedio*100:.2f}% por filmación.")


@app.get("/director/{nombre_director}")
def get_director(nombre_director: str):
    # Filtrar el DataFrame dfCreditsFinal para encontrar las películas dirigidas por el director
    peliculas_director = dfCrewFinal[
        (dfCrewFinal['name_crew'].str.contains(nombre_director, case=False, na=False)) &
        (dfCrewFinal['job_crew'].str.contains('Director', case=False, na=False))
    ]
    
    # Verificar si el director tiene películas en el dataset
    if peliculas_director.empty:
        return f"El director {nombre_director} no se encuentra en el dataset."

    # Obtener los IDs de las películas dirigidas por el director
    ids_peliculas = peliculas_director['id_pelicula'].unique()
    
    # Filtrar el DataFrame dfMoviesFinal usando los IDs de las películas del director
    peliculas_info = dfMoviesFinal[dfMoviesFinal['id_pelicula'].isin(ids_peliculas)]
    peliculas_info = peliculas_info.drop_duplicates(subset='id_pelicula')

    # Calcular el retorno y extraer la información de las películas
    peliculas_info['ganancia'] = peliculas_info['revenue'] - peliculas_info['budget']
    peliculas_info['retorno'] = peliculas_info['return']

    # Preparar la salida
    resultados = []
    for _, row in peliculas_info.iterrows():
        resultado = (
            f"Película: {row['title']}, "
            f"Fecha de lanzamiento: {row['release_date']}, "
            f"Retorno: {row['retorno']*100:.2f}%, "
            f"Costo: ${row['budget']:.2f}, "
            f"Ganancia: ${row['ganancia']:.2f}"
        )
        resultados.append(resultado)
    
    return "\n".join(resultados)


@app.get("/recomendacion/{title}")
def recomendacion(title):
    # Vectorizar los resúmenes utilizando TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(dfMoviesFinal['processed_overview'])

    # Verificar si el título existe en el DataFrame
    if title not in dfMoviesFinal['title'].values:
        raise ValueError(f"El título '{title}' no se encuentra en el DataFrame.")
    
    # Obtener el resumen preprocesado de la película dada
    processed_summary = dfMoviesFinal[dfMoviesFinal['title'] == title]['processed_overview'].values[0]
    query_vector = vectorizer.transform([processed_summary])

    # Calcular la similitud del coseno entre la película consultada y todas las demás
    cosine_sim = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Obtener los índices de las 5 películas más similares
    similar_indices = cosine_sim.argsort()[-6:-1][::-1]

    # Obtener las 5 películas más similares
    recommended_movies = dfMoviesFinal.iloc[similar_indices]
    return recommended_movies[['title', 'overview']]