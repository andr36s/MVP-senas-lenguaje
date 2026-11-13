"""
Etapa 5: servidor FastAPI.

Por qué FastAPI: valida automáticamente los datos de entrada (63 números),
genera documentación en /docs y es muy liviano.

Para qué: carga el modelo UNA sola vez al arrancar (cargarlo en cada
petición tardaría ~1 s) y expone:
    GET  /          -> la página web (index.html)
    POST /predecir  -> recibe 63 números y devuelve {letra, confianza}

El servidor solo escucha en 127.0.0.1 (ver iniciar_windows.bat): nadie más
en la red puede conectarse, y el video nunca sale del navegador.

Uso:
    python -m uvicorn api:app --host 127.0.0.1 --port 8000
"""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from comun import N_VALORES, normalizar_puntos

RAIZ = Path(__file__).resolve().parent
estado = {}


@asynccontextmanager
async def ciclo_de_vida(app):
    # Se ejecuta una vez al iniciar el servidor.
    etiquetas = json.loads((RAIZ / "etiquetas.json").read_text(encoding="utf-8"))
    estado["modelo"] = keras.saving.load_model(RAIZ / "modelo_senas.keras")
    estado["nombres"] = etiquetas["nombres"]
    estado["umbral"] = etiquetas["umbral_confianza"]
    print(f"Modelo cargado: {len(estado['nombres'])} letras, "
          f"umbral de confianza {estado['umbral']:.0%}")
    yield
    estado.clear()


app = FastAPI(title="Señas LSC", version="1.0", lifespan=ciclo_de_vida,
              description="Reconoce letras estáticas de la Lengua de Señas Colombiana")


class Puntos(BaseModel):
    # 21 puntos x (x, y, z) en píxeles, tal como los da MediaPipe multiplicados
    # por el ancho y alto del video. El servidor los normaliza.
    puntos: list[float] = Field(..., min_length=N_VALORES, max_length=N_VALORES)


class Prediccion(BaseModel):
    letra: str | None  # None cuando no está seguro
    confianza: float
    mensaje: str


@app.post("/predecir", response_model=Prediccion)
def predecir(datos: Puntos):
    vector = normalizar_puntos(datos.puntos)
    if vector is None:
        raise HTTPException(status_code=422, detail="Mano no válida (puntos superpuestos)")
    x = np.asarray([vector], dtype=np.float32)
    # Llamar al modelo directamente es más rápido que model.predict() para
    # una sola muestra (predict prepara un lote completo cada vez).
    prob = keras.ops.convert_to_numpy(estado["modelo"](x, training=False))[0]
    i = int(prob.argmax())
    confianza = float(prob[i])
    if confianza < estado["umbral"]:
        # Mejor admitir la duda que inventar una letra.
        return Prediccion(letra=None, confianza=confianza, mensaje="No estoy seguro")
    letra = estado["nombres"][i]
    return Prediccion(letra=letra, confianza=confianza, mensaje=letra)


@app.get("/salud")
def salud():
    return {"ok": "modelo" in estado, "letras": estado.get("nombres", [])}


@app.get("/", include_in_schema=False)
def pagina():
    return FileResponse(RAIZ / "index.html")
