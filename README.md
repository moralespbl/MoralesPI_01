<p align=center><img src=https://d31uz8lwfmyn8g.cloudfront.net/Assets/logo-henry-white-lg.png><p>

# <h1 align=center> **PROYECTO INDIVIDUAL Nº1** </h1>
# <h1 align=center> *Machine Learning Operations (MLOps)* </h1>

## 1. Introducion
    Este proyecto individual tiene como fin cumplir con lo solicitado por el boot camp de SoyHenry de Data Sciences, es el primero de tres que se llevaran a cabo.
    Para ello aplicaremos lo aprendidos en el curso, en este caso comenzamos con el tratamiento y recoleccion de datos hasta el entrenamiento y mantenimiento de modelos de ML.
    Creamos un modelo de recomendacion de peliculas para una plataforma de contenidos.
    Comenzamos desce cero realizando un rapido trabajo de Data Engineer y obtener un MVP.

## 2. Instalación y Requisitos
  <h3>Requisitos:</h3>
  <ul>
    <li>Python 3.10 o superior</li>
    <li>pandas</li>
    <li>json</li>
    <li>numpy</li>
    <li>matplotlib</li>
    <li>scikit-learn</li>
    <li>seaborn</li>
    <li>st</li>
    <li>nltk</li>
  </ul>
   <h3>Pasos de instalación:</h3>
    <ol>
    <li>Clonar el repositorio: <code>git clone https://github.com/moralespbl/MoralesPI_01.git</code></li>
    <li>Crear un entorno virtual: <code>python -m venv venv</code></li>
    <li>Activar el entorno virtual:
        <ul>
        <li>Windows: <code>venv\Scripts\activate</code></li>
        <li>macOS/Linux: <code>source venv/bin/activate</code></li>
        </ul>
    </li>
    <li>Instalar las dependencias: <code>pip install -r requirements.txt</code></li>
    </ol>

## 3. Estructura del Proyecto
- `DataSets/`: Contiene los archivos de datos utilizados en el modelo de recomendacion. Por cuestiones de limites de almacenamiento en git hub (100 mb) no contiene los data frames  originales.
- `DataSets/dfCastFinal.parquet`: Contiene solo los actores.
- `DataSets/dfCastCrew.parquet`: Contiene solo los directores.
- `DataSets/dfMoviesSintetico.parquet`: Contiene las peliculas en idioma Ingles
- `ETL.ipynb`: Notebooks con la limpieza de datos solocitado.
- `EDA.ipynb`: Notebooks con el analisis exploratorio de datos, ademas incluye   limpieza de datos solocitado adicionales con el fin de hacer mas livianos los archivos que se deployaran.
- `MODELO.ipynb`: Notebooks que contiene el modelo de recomendacion de datos.
- `main.py`: Código fuente del proyecto, incluyendo scripts y módulos.
- `requirements.txt`: Guarda los informes y visualizaciones generados.
- `README.md`: Archivo de documentación del proyecto.

## 4. Uso y Ejecución
1. Para ejecutar el ETL, abrir el notebook `ETL.ipynb`, los data frames crudos no se encuentran almacenados en github. El notebook guiará a través de las diferentes etapas cumpliendo lo solicitado por las consignas.
2. Para ejecutar el EDA, abrir el notebook `EDA.ipynb`, los data frames crudos no se encuentran almacenados en github. En notebook encontrara el analisis de datos realizado y obtendra como resultado tres Data Frames. Uno de peliculas, otro de Actores y uno mas de Directores
3. Para ejecutar el MODELO, abrir el notebook `MODELO.ipynb`, contiene el modelo de machine del modelo de recomendacion y el resultado final es el data sets de peliculas que sera utilizados en la API.
4. Para generar utilizar la API debera ingresar a: [API](https://moralespi-01.onrender.com/docs)
alli encontrara las funciones solicitadas en la consigna.  
Debera tener en cuenta que para la funcion Recomendacion, se ingresara como parametro el nombre de la pelicula en idioma Ingles y la primer letra de cada palabra con Mayuscula y el resto en minuscula.

## 5. Datos y Fuentes
Para ingresar a los [Datos](https://drive.google.com/drive/folders/1X_LdCoGTHJDbD28_dJTxaD4fVuQC9Wt5?usp=drive_link) crudos.

## 6. Metodología
Una vez descargados los datos nos encontramos con dos Data Frames. Se aplicaron los pasos solicitados en las consigas en el ETL.
Para el EDA, se procedio a detectar los Outliers con un diagrama de de caja y bigotes, estos fueron reemplazados por los limites respectivos, bajo el supuesto que la base de datos no contenian valores erroneos, sino solamente extremos.
A continuacion los datos fueron escalados, utilizando un proceso de standar, por tener diferentes escalas y unidades de medida.
Una vez estandarizados, se crearon mapas de calor para estudiar la covavianza y comprender la relacion entre ellos.
Tambien se construyo un pair plot, utilizando como variable categorica los generos, y poder detectar si estos generos categorizaban correctamente a los datos.
De estos analisis se desprende que existe baja correlacion de datos y una gran dispercio de datos, como tambien que los generos no agrupa a peliculas similares entre ellas.
Se eliminaron mas columnas con el fin de ocupar menos memoria en github. 
Y si eliminaron las peliculas que no fueran de habla inglesa.
Dejando estas por ser las puntuadas.
En cuanto al modelo de recomendacion de peliculas, se utilizo el metodo de similitud del coseno en combinacion con TF-IDF para vectorizacion de texto.
En una primera instancia se preprocesaron los comentarios de las peliculas. Tockenizando, eliminado las palabras poco relevantes y lemantizando para trabajar con las raises de las palabras.
Este preprocesamiento genero un nueva columna la cual reemplazo a la columna de los comentarios en el data frame de peliculas. Siendo este el definitivo y que es usado en la API.

## 7. Resultados y Conclusiones
Se cumplieron con las consignas solicitadas.
La mayor correlacion entre variables se da entre el conteo de votos y los ingresos por peliculas (0.81) seguid por el conteo de votos y la popularidad (0.70)
La distribucion de los votos tiene forma de una distribucion normal.
El genero con mas peliculas es el Drama
El data frame tiene 45346 votos, siendo la media 5.62 y el desvio estandar de 1.91 en tanto la media es de 6.
En el pair plot se puede aprecia que los votos no se encuentran correlacionas con ninguna otra variable numerica.

## 8. Autor:
Este proyecto fue realizado por Pablo Alberto Morales Valeriano

## 9. Links Entregables:
- [API](https://moralespi-01.onrender.com/docs)
- [Repo de github](https://github.com/moralespbl/MoralesPI_01.git)
- [Video](https://1drv.ms/v/s!Aj2cX0aTkbryhqFF2-OH_zoRHDrVwQ?e=N0UShB)