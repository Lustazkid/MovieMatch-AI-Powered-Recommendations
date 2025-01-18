# Importar Librerías y Cargar los Datos

import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nest_asyncio
from uvicorn import run

# Importe y carga de datos
df_movies = pd.read_csv("movies_dataset.csv", low_memory=False)
df_credits = pd.read_csv("credits.csv")

# Exploración Inicial de los Datos (EDA)

# Exploración Inicial
print("Primeras filas de df_movies:")
print(df_movies.head())

print("\nPrimeras filas de df_credits:")
print(df_credits.head())

# Información de los Data frame
print("\nInformación de df_movies:")
print(df_movies.info())

print("\nInformación de df_credits:")
print(df_credits.info())

# Reducción df

# Recorte aleatorio del 50% del DataFrame 'credits.csv'
df_credits = df_credits.sample(frac=0.5, random_state=42)
df_credits.to_csv("credits.csv", index=False)

# Transformación de Datos (ETL) (Desanidar Columnas y Llenar Valores Nulos)

# Desanidar columnas específicas
def desanidar_json(json_column):
    if isinstance(json_column, str):
        try:
            return eval(json_column)
        except:
            return np.nan
    return json_column

df_movies['belongs_to_collection'] = df_movies['belongs_to_collection'].apply(desanidar_json)
df_movies['production_companies'] = df_movies['production_companies'].apply(desanidar_json)

# Rellenar valores nulos en 'revenue' y 'budget' con 0
df_movies['revenue'] = df_movies['revenue'].fillna(0)
df_movies['budget'] = df_movies['budget'].fillna(0)

# Formatear y limpiar columnas
df_movies['revenue'] = pd.to_numeric(df_movies['revenue'], errors='coerce').fillna(0)
df_movies['budget'] = pd.to_numeric(df_movies['budget'], errors='coerce').fillna(0)

# Eliminar valores nulos en 'release_date'
df_movies = df_movies.dropna(subset=['release_date']).copy()

# Formatear la columna 'release_date' al formato AAAA-mm-dd
df_movies['release_date'] = pd.to_datetime(df_movies['release_date'], errors='coerce').dt.strftime('%Y-%m-%d')

# Crear la columna 'release_year'
df_movies['release_year'] = pd.to_datetime(df_movies['release_date']).dt.year

# Crear la columna 'return' (retorno de inversión)
df_movies['return'] = df_movies.apply(
    lambda x: x['revenue'] / x['budget'] if x['budget'] > 0 else 0, axis=1
)

# Eliminar columnas que no serán utilizadas
columns_to_drop = ['video', 'imdb_id', 'adult', 'original_title', 'poster_path', 'homepage']
df_movies = df_movies.drop(columns=columns_to_drop, errors='ignore').copy()

# Guardar el dataframe transformado
df_movies.to_csv("transformed_movies.csv", index=False)

# Verificar columnas disponibles antes de intentar eliminarlas
print("Columnas actuales del DataFrame:", df_movies.columns)

# Eliminar columnas innecesarias (ignorar errores si alguna no existe)
columns_to_drop = ['video', 'imdb_id', 'adult', 'original_title', 'poster_path', 'homepage']
df_movies = df_movies.drop(columns=columns_to_drop, errors='ignore')

# Convertir 'id' a numérico y realizar el merge con df_credits
df_credits['id'] = pd.to_numeric(df_credits['id'], errors='coerce')
df_movies['id'] = pd.to_numeric(df_movies['id'], errors='coerce')

# Realizar el merge
df_merged = pd.merge(df_movies, df_credits, on='id', how='left')

# Implementación de la API (Configuración de la API)

# Cargar el archivo
df = pd.read_csv("movies_dataset.csv", low_memory=False)

# Llenar valores nulos en la columna 'overview' (importante para TF-IDF)
df['overview'] = df['overview'].fillna('')
# Crear el vectorizador TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Generar la matriz TF-IDF con la columna 'overview'
tfidf_matrix = tfidf_vectorizer.fit_transform(df['overview'])

# Cargar el dataset transformado
df = pd.read_csv("transformed_movies.csv", low_memory=False)

# Crear la aplicación FastAPI
app = FastAPI()

# Habilitar CORS (opcional)
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoints de la API

@app.get("/")
def home():
    return {"message": "Bienvenido a la API de Películas"}

@app.get("/cantidad_filmaciones_mes/{mes}")
def cantidad_filmaciones_mes(mes: str):
    meses = {
        "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
        "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
        "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12
    }
    mes_num = meses.get(mes.lower())
    if mes_num:
        cantidad = df[pd.to_datetime(df['release_date']).dt.month == mes_num].shape[0]
        return {"mes": mes, "cantidad": cantidad}
    return {"error": "Mes inválido"}

@app.get("/cantidad_filmaciones_dia/{dia}")
def cantidad_filmaciones_dia(dia: str):
    dias = {
        "lunes": 0, "martes": 1, "miércoles": 2, "jueves": 3,
        "viernes": 4, "sábado": 5, "domingo": 6
    }
    dia_num = dias.get(dia.lower())
    if dia_num is not None:
        cantidad = df[pd.to_datetime(df['release_date']).dt.weekday == dia_num].shape[0]
        return {"dia": dia, "cantidad": cantidad}
    return {"error": "Día inválido"}

@app.get("/score_titulo/{titulo}")
def score_titulo(titulo: str):
    resultado = df[df['title'].str.contains(titulo, case=False, na=False)]
    if not resultado.empty:
        pelicula = resultado.iloc[0]
        return {
            "titulo": pelicula['title'],
            "año": pelicula['release_year'],
            "score": pelicula['popularity']
        }
    return {"error": "Película no encontrada"}

@app.get("/votos_titulo/{titulo}")
def votos_titulo(titulo: str):
    resultado = df[df['title'].str.contains(titulo, case=False, na=False)]
    if not resultado.empty:
        pelicula = resultado.iloc[0]
        if pelicula['vote_count'] >= 2000:
            return {
                "titulo": pelicula['title'],
                "cantidad_votos": pelicula['vote_count'],
                "promedio_votos": pelicula['vote_average']
            }
        return {"error": "La película no tiene suficientes votos"}
    return {"error": "Película no encontrada"}

@app.get("/get_actor/{nombre_actor}")
def get_actor(nombre_actor: str):
    actores = df[df['cast'].str.contains(nombre_actor, case=False, na=False)]
    if not actores.empty:
        retorno_total = actores['return'].sum()
        cantidad = actores.shape[0]
        promedio = retorno_total / cantidad if cantidad > 0 else 0
        return {
            "actor": nombre_actor,
            "cantidad_filmaciones": cantidad,
            "retorno_total": retorno_total,
            "promedio_retorno": promedio
        }
    return {"error": "Actor no encontrado"}

@app.get("/get_director/{nombre_director}")
def get_director(nombre_director: str):
    directores = df[df['crew'].str.contains(nombre_director, case=False, na=False)]
    if not directores.empty:
        peliculas = []
        for _, row in directores.iterrows():
            peliculas.append({
                "titulo": row['title'],
                "fecha_lanzamiento": row['release_date'],
                "retorno": row['return'],
                "costo": row['budget'],
                "ganancia": row['revenue']
            })
        retorno_total = directores['return'].sum()
        return {
            "director": nombre_director,
            "retorno_total": retorno_total,
            "peliculas": peliculas
        }
    return {"error": "Director no encontrado"}
@app.get("/recomendaciones/{titulo}")
def recomendaciones(titulo: str, n_recomendaciones: int = 5):
    try:
        # Asegúrate de que el título está en el dataset
        if titulo not in df['title'].values:
            return JSONResponse(
                content={"error": f"El título '{titulo}' no se encuentra en el dataset."},
                status_code=404
            )
        
        # Obtén el índice de la película
        idx = df[df['title'] == titulo].index[0]
        
        # Calcula similitudes
        similitudes = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        
        # Obtén índices de películas más similares
        indices_similares = similitudes.argsort()[::-1][1:n_recomendaciones + 1]
        
        # Genera la respuesta
        recomendaciones = df.iloc[indices_similares][['title', 'overview']].to_dict(orient='records')
        return {"titulo": titulo, "recomendaciones": recomendaciones}
    
    except Exception as e:
        # Manejo de errores generales
        return JSONResponse(
            content={"error": f"Error procesando la solicitud: {str(e)}"},
            status_code=500
        )
# Ejecución Local

nest_asyncio.apply()

# Ejecutar la aplicación
run(app, host="0.0.0.0", port=8000)
