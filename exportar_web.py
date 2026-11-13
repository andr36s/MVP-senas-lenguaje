"""
Exporta el modelo para que la app funcione sin servidor (GitHub Pages).

Por qué: para probar la app desde el celular fuera de casa hace falta una
dirección HTTPS pública (la cámara del celular solo funciona en páginas
seguras). Abrir el computador a internet es un riesgo; en cambio, una página
estática en GitHub Pages es gratis, tiene HTTPS y no expone el computador.

Para qué: el modelo entrenado con Keras es una red de 3 capas Dense, así que
predecir es solo multiplicar matrices: y = activación(x · W + b). Este script
guarda W y b de cada capa en docs/modelo.json y copia index.html a docs/.
Cuando index.html encuentra modelo.json a su lado, hace esas multiplicaciones
en JavaScript dentro del celular; si no lo encuentra, consulta a api.py como
siempre. El entrenamiento sigue siendo 100 % Keras.

Uso (después de entrenar.py):
    python exportar_web.py
"""
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras

RAIZ = Path(__file__).resolve().parent
DESTINO = RAIZ / "docs"  # GitHub Pages puede publicar la carpeta /docs de main


def main():
    modelo = keras.saving.load_model(RAIZ / "modelo_senas.keras")
    etiquetas = json.loads((RAIZ / "etiquetas.json").read_text(encoding="utf-8"))

    capas = []
    for capa in modelo.layers:
        if not isinstance(capa, keras.layers.Dense):
            continue  # Dropout no tiene pesos y al predecir no hace nada
        w, b = capa.get_weights()  # w: (entradas, salidas)
        capas.append({
            "activacion": capa.get_config()["activation"],
            "W": [[round(float(v), 7) for v in fila] for fila in w],
            "b": [round(float(v), 7) for v in b],
        })
    datos = {
        "letras": etiquetas["nombres"],
        "umbral_confianza": etiquetas["umbral_confianza"],
        "capas": capas,
    }

    DESTINO.mkdir(exist_ok=True)
    (DESTINO / "modelo.json").write_text(json.dumps(datos, separators=(",", ":")),
                                         encoding="utf-8")
    shutil.copyfile(RAIZ / "index.html", DESTINO / "index.html")
    tamano = (DESTINO / "modelo.json").stat().st_size / 1024
    print(f"Exportadas {len(capas)} capas Dense ({tamano:.0f} KB) y index.html en {DESTINO}")


if __name__ == "__main__":
    main()
