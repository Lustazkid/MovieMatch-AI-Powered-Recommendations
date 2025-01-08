{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "1. Importar Librerías y Cargar los Datos\n",
    "(Primera Celda)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Importar librerías necesarias\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from fastapi import FastAPI\n",
    "from fastapi.responses import JSONResponse\n",
    "from fastapi.middleware.cors import CORSMiddleware\n",
    "from pydantic import BaseModel\n",
    "import nest_asyncio\n",
    "from uvicorn import run\n",
    "\n",
    "# Cargar los datasets\n",
    "df_movies = pd.read_csv(\"movies_dataset.csv\", low_memory=False)\n",
    "df_credits = pd.read_csv(\"credits.csv\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "2. Exploración Inicial de los Datos (EDA)\n",
    "(Segunda Celda)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Mostrar las primeras filas de los datasets\n",
    "print(\"Primeras filas de df_movies:\")\n",
    "print(df_movies.head())\n",
    "\n",
    "print(\"\\nPrimeras filas de df_credits:\")\n",
    "print(df_credits.head())\n",
    "\n",
    "# Información general de las columnas y tipos de datos\n",
    "print(\"\\nInformación de df_movies:\")\n",
    "print(df_movies.info())\n",
    "\n",
    "print(\"\\nInformación de df_credits:\")\n",
    "print(df_credits.info())\n",
    "\n",
    "# Listar columnas de cada dataset\n",
    "print(\"\\nColumnas en df_movies:\")\n",
    "print(df_movies.columns)\n",
    "\n",
    "print(\"\\nColumnas en df_credits:\")\n",
    "print(df_credits.columns)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Tomar una muestra aleatoria del 50% del DataFrame de 'credits.csv'\n",
    "df_credits = df_credits.sample(frac=0.5, random_state=42)\n",
    "\n",
    "# Sobrescribir el archivo CSV original con la muestra aleatoria\n",
    "df_credits.to_csv(\"credits.csv\", index=False)\n",
    "\n",
    "# Verificar las primeras filas para confirmar\n",
    "print(df_credits.info())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "3. Transformación de Datos (ETL)\n",
    "(Tercera Celda - Desanidar Columnas y Llenar Valores Nulos)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Desanidar columnas específicas\n",
    "def desanidar_json(json_column):\n",
    "    if isinstance(json_column, str):\n",
    "        try:\n",
    "            return eval(json_column)  # Evaluar como JSON si es posible\n",
    "        except:\n",
    "            return np.nan\n",
    "    return json_column\n",
    "\n",
    "df_movies['belongs_to_collection'] = df_movies['belongs_to_collection'].apply(desanidar_json)\n",
    "df_movies['production_companies'] = df_movies['production_companies'].apply(desanidar_json)\n",
    "\n",
    "# Rellenar valores nulos en 'revenue' y 'budget' con 0\n",
    "df_movies['revenue'] = df_movies['revenue'].fillna(0)\n",
    "df_movies['budget'] = df_movies['budget'].fillna(0)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "(Cuarta Celda - Transformación de Fechas y Columnas Nuevas)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Desanidar algunos campos (opcional según lo necesites)\n",
    "\n",
    "# 2. Rellenar valores nulos en 'revenue' y 'budget' con 0\n",
    "df_movies['revenue'] = pd.to_numeric(df_movies['revenue'], errors='coerce').fillna(0)\n",
    "df_movies['budget'] = pd.to_numeric(df_movies['budget'], errors='coerce').fillna(0)\n",
    "\n",
    "# 3. Eliminar valores nulos en 'release_date'\n",
    "df_movies = df_movies.dropna(subset=['release_date']).copy()\n",
    "\n",
    "# 4. Formatear la columna 'release_date' al formato AAAA-mm-dd\n",
    "df_movies.loc[:, 'release_date'] = pd.to_datetime(df_movies['release_date'], errors='coerce').dt.strftime('%Y-%m-%d')\n",
    "\n",
    "# 5. Crear la columna 'release_year' extrayendo el año de la fecha de estreno\n",
    "df_movies.loc[:, 'release_year'] = pd.to_datetime(df_movies['release_date']).dt.year\n",
    "\n",
    "# 6. Crear la columna 'return' (retorno de inversión)\n",
    "df_movies.loc[:, 'return'] = df_movies.apply(\n",
    "    lambda x: x['revenue'] / x['budget'] if x['budget'] > 0 else 0, axis=1\n",
    ")\n",
    "\n",
    "# 7. Eliminar columnas que no serán utilizadas\n",
    "columns_to_drop = ['video', 'imdb_id', 'adult', 'original_title', 'poster_path', 'homepage']\n",
    "df_movies = df_movies.drop(columns=columns_to_drop, errors='ignore').copy()\n",
    "\n",
    "# Verificar cambios\n",
    "print(\"\\nPrimeras filas después de las transformaciones:\")\n",
    "print(df_movies.head())\n",
    "\n",
    "print(\"\\nInformación del DataFrame después de las transformaciones:\")\n",
    "print(df_movies.info())\n",
    "\n",
    "df_movies.to_csv(\"transformed_movies.csv\", index=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "(Quinta Celda - Eliminar Columnas y Realizar el Merge)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Verificar columnas disponibles antes de intentar eliminarlas\n",
    "print(\"Columnas actuales del DataFrame:\", df_movies.columns)\n",
    "\n",
    "# Eliminar columnas innecesarias (ignorar errores si alguna no existe)\n",
    "columns_to_drop = ['video', 'imdb_id', 'adult', 'original_title', 'poster_path', 'homepage']\n",
    "df_movies = df_movies.drop(columns=columns_to_drop, errors='ignore')\n",
    "\n",
    "# Convertir 'id' a numérico y realizar el merge con df_credits\n",
    "df_credits['id'] = pd.to_numeric(df_credits['id'], errors='coerce')\n",
    "df_movies['id'] = pd.to_numeric(df_movies['id'], errors='coerce')\n",
    "\n",
    "# Realizar el merge\n",
    "df_merged = pd.merge(df_movies, df_credits, on='id', how='left')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "4. Implementación de la API\n",
    "(Sexta Celda - Configuración de la API)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cargar el dataset transformado\n",
    "df = pd.read_csv(\"transformed_movies.csv\", low_memory=False)\n",
    "# Crear la aplicación FastAPI\n",
    "app = FastAPI()\n",
    "\n",
    "# Habilitar CORS (opcional)\n",
    "origins = [\"*\"]\n",
    "app.add_middleware(\n",
    "    CORSMiddleware,\n",
    "    allow_origins=origins,\n",
    "    allow_credentials=True,\n",
    "    allow_methods=[\"*\"],\n",
    "    allow_headers=[\"*\"],\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "(Séptima Celda - Endpoints de la API)\n",
    "python\n",
    "Copiar código\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "@app.get(\"/\")\n",
    "def home():\n",
    "    return {\"message\": \"Bienvenido a la API de Películas\"}\n",
    "\n",
    "@app.get(\"/cantidad_filmaciones_mes/{mes}\")\n",
    "def cantidad_filmaciones_mes(mes: str):\n",
    "    meses = {\n",
    "        \"enero\": 1, \"febrero\": 2, \"marzo\": 3, \"abril\": 4,\n",
    "        \"mayo\": 5, \"junio\": 6, \"julio\": 7, \"agosto\": 8,\n",
    "        \"septiembre\": 9, \"octubre\": 10, \"noviembre\": 11, \"diciembre\": 12\n",
    "    }\n",
    "    mes_num = meses.get(mes.lower())\n",
    "    if mes_num:\n",
    "        cantidad = df[pd.to_datetime(df['release_date']).dt.month == mes_num].shape[0]\n",
    "        return {\"mes\": mes, \"cantidad\": cantidad}\n",
    "    return {\"error\": \"Mes inválido\"}\n",
    "\n",
    "@app.get(\"/cantidad_filmaciones_dia/{dia}\")\n",
    "def cantidad_filmaciones_dia(dia: str):\n",
    "    dias = {\n",
    "        \"lunes\": 0, \"martes\": 1, \"miércoles\": 2, \"jueves\": 3,\n",
    "        \"viernes\": 4, \"sábado\": 5, \"domingo\": 6\n",
    "    }\n",
    "    dia_num = dias.get(dia.lower())\n",
    "    if dia_num is not None:\n",
    "        cantidad = df[pd.to_datetime(df['release_date']).dt.weekday == dia_num].shape[0]\n",
    "        return {\"dia\": dia, \"cantidad\": cantidad}\n",
    "    return {\"error\": \"Día inválido\"}\n",
    "\n",
    "@app.get(\"/score_titulo/{titulo}\")\n",
    "def score_titulo(titulo: str):\n",
    "    resultado = df[df['title'].str.contains(titulo, case=False, na=False)]\n",
    "    if not resultado.empty:\n",
    "        pelicula = resultado.iloc[0]\n",
    "        return {\n",
    "            \"titulo\": pelicula['title'],\n",
    "            \"año\": pelicula['release_year'],\n",
    "            \"score\": pelicula['popularity']\n",
    "        }\n",
    "    return {\"error\": \"Película no encontrada\"}\n",
    "\n",
    "@app.get(\"/votos_titulo/{titulo}\")\n",
    "def votos_titulo(titulo: str):\n",
    "    resultado = df[df['title'].str.contains(titulo, case=False, na=False)]\n",
    "    if not resultado.empty:\n",
    "        pelicula = resultado.iloc[0]\n",
    "        if pelicula['vote_count'] >= 2000:\n",
    "            return {\n",
    "                \"titulo\": pelicula['title'],\n",
    "                \"cantidad_votos\": pelicula['vote_count'],\n",
    "                \"promedio_votos\": pelicula['vote_average']\n",
    "            }\n",
    "        return {\"error\": \"La película no tiene suficientes votos\"}\n",
    "    return {\"error\": \"Película no encontrada\"}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "(Octava Celda - Continuación de Endpoints)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "@app.get(\"/get_actor/{nombre_actor}\")\n",
    "def get_actor(nombre_actor: str):\n",
    "    actores = df[df['cast'].str.contains(nombre_actor, case=False, na=False)]\n",
    "    if not actores.empty:\n",
    "        retorno_total = actores['return'].sum()\n",
    "        cantidad = actores.shape[0]\n",
    "        promedio = retorno_total / cantidad if cantidad > 0 else 0\n",
    "        return {\n",
    "            \"actor\": nombre_actor,\n",
    "            \"cantidad_filmaciones\": cantidad,\n",
    "            \"retorno_total\": retorno_total,\n",
    "            \"promedio_retorno\": promedio\n",
    "        }\n",
    "    return {\"error\": \"Actor no encontrado\"}\n",
    "\n",
    "@app.get(\"/get_director/{nombre_director}\")\n",
    "def get_director(nombre_director: str):\n",
    "    directores = df[df['crew'].str.contains(nombre_director, case=False, na=False)]\n",
    "    if not directores.empty:\n",
    "        peliculas = []\n",
    "        for _, row in directores.iterrows():\n",
    "            peliculas.append({\n",
    "                \"titulo\": row['title'],\n",
    "                \"fecha_lanzamiento\": row['release_date'],\n",
    "                \"retorno\": row['return'],\n",
    "                \"costo\": row['budget'],\n",
    "                \"ganancia\": row['revenue']\n",
    "            })\n",
    "        retorno_total = directores['return'].sum()\n",
    "        return {\n",
    "            \"director\": nombre_director,\n",
    "            \"retorno_total\": retorno_total,\n",
    "            \"peliculas\": peliculas\n",
    "        }\n",
    "    return {\"error\": \"Director no encontrado\"}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "5. Ejecución Local\n",
    "(Novena Celda)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Usar nest_asyncio para permitir la ejecución en Jupyter Notebook\n",
    "nest_asyncio.apply()\n",
    "\n",
    "# Ejecutar la aplicación\n",
    "run(app, host=\"0.0.0.0\", port=8000)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
