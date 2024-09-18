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
    """
    Retorna la cantidad de películas únicas que fueron estrenadas en un mes específico.

    Parámetros:
    mes (str): El nombre del mes en español (debe ser en minúsculas o mayúsculas).

    Retorna:
    str: Un mensaje indicando la cantidad de películas estrenadas en el mes solicitado.

    Lanza:
    ValueError: Si el valor de 'mes' no es válido (no corresponde a un mes en español).
    """
    
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
def cantidad_filmaciones_dia(dia: str):
    """
    Retorna la cantidad de películas únicas que fueron estrenadas en un día específico de la semana.

    Parámetros:
    dia (str): El nombre del día de la semana en español (debe ser en minúsculas o mayúsculas).

    Retorna:
    str: Un mensaje indicando la cantidad de películas estrenadas en el día solicitado.

    Lanza:
    ValueError: Si el valor de 'dia' no es válido (no corresponde a un día de la semana en español).
    """
    
    dias_dict = {
        'lunes': 1, 'martes': 2, 'miercoles': 3, 'jueves': 4, 'viernes': 5, 'sabado': 6, 'domingo': 7
    }
    dia = dia.lower()

    if dia not in dias_dict:
        raise ValueError("Día no válido. Debe ser uno de los siguientes: lunes, martes, miércoles, jueves, viernes, sábado, domingo.")
    
    dia_numero = dias_dict[dia]

    cantidad = dfMoviesFinal[dfMoviesFinal['release_date'].dt.dayofweek + 1 == dia_numero]['id_pelicula'].nunique()
    
    return f"{cantidad} películas fueron estrenadas en los días {dia}"




@app.get("/peliculas/score/{titulo}")
@app.get("/peliculas/score/{titulo}")
def score_titulo(nombre_peli: str):
    """
    Busca la película por título y devuelve su año de estreno y puntuación de popularidad (score).

    Parámetros:
    nombre_peli (str): El nombre (o parte del nombre) de la película a buscar.

    Retorna:
    str: Un mensaje con el título exacto de la película, el año de estreno y su puntuación.
         Si no se encuentra la película, devuelve un mensaje indicando que no fue encontrada.
    """
    
    pelicula = dfMoviesFinal[dfMoviesFinal['title'].str.contains(nombre_peli, case=False, na=False)]

    if pelicula.empty:
        return f"No se encontró la película '{nombre_peli}'."

    titulo = pelicula.iloc[0]['title']
    anio_estreno = pelicula.iloc[0]['release_year']
    puntuacion = pelicula.iloc[0]['vote_average']

    return f"La película '{titulo}' fue estrenada en el año {anio_estreno} con un score/popularidad de {puntuacion}."



@app.get("/peliculas/votos/{titulo}")
def votos_titulo(titulo_de_la_filmacion: str):
    """
    Busca una película por su título y devuelve la cantidad de votos y el promedio de votos si cumple con el mínimo de 2000 valoraciones.

    Parámetros:
    titulo_de_la_filmacion (str): El nombre (o parte del nombre) de la película a buscar.

    Retorna:
    str: Un mensaje con el título exacto de la película, el año de estreno, la cantidad de votos y el promedio de votos.
         Si la película no tiene al menos 2000 valoraciones o no se encuentra, devuelve un mensaje indicando la situación.
    """

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
    """
    Busca al actor en el DataFrame de películas y devuelve información sobre la cantidad de películas en las que ha participado,
    así como el retorno total y promedio de esas películas.

    Parámetros:
    nombre_actor (str): El nombre (o parte del nombre) del actor a buscar.

    Retorna:
    str: Un mensaje con la cantidad de películas en las que ha actuado el actor, el retorno total y el retorno promedio.
         Si el actor no aparece en el dataset o no tiene películas con presupuesto válido, devuelve un mensaje indicando la situación.
    """

    peliculas_actor = dfCastFinal[dfCastFinal['name_cast'].str.contains(nombre_actor, case=False, na=False)]

    if peliculas_actor.empty:
        return f"No se encontró al actor {nombre_actor} en el dataset."

    ids_peliculas = peliculas_actor['id_pelicula'].drop_duplicates()

    peliculas_info = dfMoviesFinal[dfMoviesFinal['id_pelicula'].isin(ids_peliculas)]
    peliculas_info = peliculas_info.drop_duplicates(subset='id_pelicula')

    retorno_total = peliculas_info['return'].sum()
    retorno_promedio = retorno_total/len(ids_peliculas)

    cantidad_peliculas = len(ids_peliculas)
    if cantidad_peliculas == 0:
        return f"El actor {nombre_actor} no tiene películas con presupuesto válido."

    return (f"El actor {nombre_actor} ha participado en {cantidad_peliculas} películas, "
            f"con un retorno total de {retorno_total*100:.2f}% y un promedio de {retorno_promedio*100:.2f}% por filmación.")


@app.get("/director/{nombre_director}")
def get_director(nombre_director: str):
    """
    Busca a un director en el DataFrame de créditos y devuelve información sobre las películas que ha dirigido,
    incluyendo el retorno, el costo y la ganancia de cada película.

    Parámetros:
    nombre_director (str): El nombre (o parte del nombre) del director a buscar.

    Retorna:
    str: Un mensaje con información de cada película dirigida por el director, mostrando el título, la fecha de lanzamiento,
         el retorno, el costo y la ganancia. Si el director no está en el dataset, se devuelve un mensaje indicando la situación.
    """

    peliculas_director = dfCrewFinal[
        (dfCrewFinal['name_crew'].str.contains(nombre_director, case=False, na=False)) &
        (dfCrewFinal['job_crew'].str.contains('Director', case=False, na=False))
    ]
    
    if peliculas_director.empty:
        return f"El director {nombre_director} no se encuentra en el dataset."

    ids_peliculas = peliculas_director['id_pelicula'].unique()
    
    peliculas_info = dfMoviesFinal[dfMoviesFinal['id_pelicula'].isin(ids_peliculas)]
    peliculas_info = peliculas_info.drop_duplicates(subset='id_pelicula')

    peliculas_info['ganancia'] = peliculas_info['revenue'] - peliculas_info['budget']
    peliculas_info['retorno'] = peliculas_info['return']

    for _, row in peliculas_info.iterrows():
        print (
            f"Película: {row['title']}, "
            f"Fecha de lanzamiento: {row['release_date']}, "
            f"Retorno: {row['retorno']*100:.2f}%, "
            f"Costo: ${row['budget']:.2f}, "
            f"Ganancia: ${row['ganancia']:.2f}"
        )


@app.get("/recomendacion/{title}")
def recomendacion(title):
    """
    Genera recomendaciones de películas basadas en la similitud de resúmenes de películas (sinopsis) mediante TF-IDF y similitud del coseno.

    Parámetros:
    title (str): El título de la película para la cual se desea obtener recomendaciones.

    Retorna:
    pandas.Series: Una lista con los títulos de las 5 películas más similares en términos de su sinopsis.
    
    Lanza:
    ValueError: Si el título de la película no se encuentra en el DataFrame.

    """

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
    return recommended_movies['title']