"""
main_api.py
API REST (FastAPI + Pydantic) que expone el modelo de scouting serializado.

Ejecutar desde la carpeta del proyecto (donde está la carpeta models/):
    uvicorn main_api:app --reload

Documentación interactiva (Swagger):  http://127.0.0.1:8000/docs
"""
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

RUTA_PAQUETE = "models/modelo_scouting_rf.joblib"

app = FastAPI(
    title="API de Scouting Deportivo",
    description="Predice si un jugador es Élite (Top 25% de valor de mercado) "
                "a partir de su posición, edad y estatura.",
    version="1.0.0",
)

# CORS: permite que index.html (otro origen) consuma la API con fetch
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# El paquete se carga UNA sola vez al iniciar el servidor
paquete = joblib.load(RUTA_PAQUETE)
modelo = paquete["modelo"]
columnas_entrenamiento = paquete["columnas_X"]
posiciones_validas = paquete["posiciones_validas"]


# ---------------------------------------------------------------
# Esquemas Pydantic (validación de entrada y salida)
# ---------------------------------------------------------------
class Jugador(BaseModel):
    jugador: str = Field("Sin nombre", description="Nombre del candidato")
    posicion: str = Field(..., description="Posición del jugador, p. ej. Defender")
    edad: int = Field(..., ge=14, le=50, description="Edad en años")
    estatura_cm: Optional[float] = Field(
        None, ge=120, le=230,
        description="Estatura en cm (opcional: si falta se imputa con la mediana de su posición)",
    )


class Prediccion(BaseModel):
    jugador: str
    posicion: str
    edad: int
    estatura_cm_usada: float
    prediccion: str
    probabilidad_elite: float


# ---------------------------------------------------------------
# Preprocesamiento: MISMAS reglas aprendidas en entrenamiento
# ---------------------------------------------------------------
def preprocesar(datos: Jugador) -> pd.DataFrame:
    df = pd.DataFrame([{
        "posicion": datos.posicion,
        "estatura_cm": datos.estatura_cm,
        "edad": datos.edad,
    }])

    df["estatura_cm"] = pd.to_numeric(df["estatura_cm"], errors="coerce")
    df["estatura_cm"] = df["estatura_cm"].fillna(df["posicion"].map(paquete["medianas_estatura"]))
    df["estatura_cm"] = df["estatura_cm"].fillna(paquete["mediana_global_estatura"])
    df["edad"] = df["edad"].fillna(paquete["mediana_global_edad"])

    # One-hot SIN drop_first: así cada posición conserva su columna y el reindex
    # descarta la que el modelo no usa (la categoría de referencia).
    dummies = pd.get_dummies(df[["posicion"]], dtype=int)
    X = pd.concat([df[["edad", "estatura_cm"]], dummies], axis=1)
    return X.reindex(columns=columnas_entrenamiento, fill_value=0)


# ---------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------
@app.get("/")
def estado():
    return {
        "servicio": "API de Scouting Deportivo",
        "estado": "activo",
        "umbral_elite_eur": float(paquete["umbral_elite_eur"]),
    }


@app.get("/posiciones")
def listar_posiciones():
    """Posiciones que el modelo conoce (para llenar el formulario web)."""
    return {"posiciones": posiciones_validas}


@app.post("/predict", response_model=Prediccion)
def predecir(datos: Jugador):
    if datos.posicion not in posiciones_validas:
        raise HTTPException(
            status_code=422,
            detail=f"Posición no válida. Opciones: {posiciones_validas}",
        )

    X_nuevo = preprocesar(datos)
    clase = int(modelo.predict(X_nuevo)[0])
    prob = float(modelo.predict_proba(X_nuevo)[0, 1])

    return Prediccion(
        jugador=datos.jugador,
        posicion=datos.posicion,
        edad=datos.edad,
        estatura_cm_usada=round(float(X_nuevo["estatura_cm"].iloc[0]), 1),
        prediccion="Élite" if clase == 1 else "Estándar",
        probabilidad_elite=round(prob * 100, 2),
    )