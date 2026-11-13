"""
Etapa 2: extraer los puntos de la mano de las fotos del dataset LSC70ANH.

Por qué: no entrenamos con las fotos, sino con los 21 puntos de la mano que
detecta MediaPipe. Así el modelo aprende la FORMA de la mano y no el fondo,
la luz, la ropa o el tono de piel de los voluntarios del dataset.

Para qué: produce datos/puntos_lsc70.csv, una fila por foto con
voluntario, letra, fotograma y los 63 valores normalizados.

Uso:
    python extraer_puntos.py
    python extraer_puntos.py --muestra 300      (prueba rápida)
"""
import argparse
import csv
import random
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from comun import (COLUMNAS_PUNTOS, importar_mediapipe, landmarks_a_pixeles,
                   normalizar_puntos)

RAIZ = Path(__file__).resolve().parent

# Solo letras del abecedario (27, incluida la Ñ, que el dataset llama "NN").
# Los números, MIL y MILLON quedan fuera del MVP.
LETRAS = list("ABCDEFGHIJKLMN") + ["NN"] + list("OPQRSTUVWXYZ")


def preparar_imagen(img, escala, margen):
    """
    Las fotos son de 120x120 y la mano llena casi todo el recuadro. El
    detector de palmas de MediaPipe funciona peor con manos tan pequeñas en
    píxeles y tan pegadas al borde, así que (1) ampliamos la imagen y
    (2) le agregamos un margen replicando el borde, para darle contexto.
    """
    if escala != 1:
        img = cv2.resize(img, None, fx=escala, fy=escala,
                         interpolation=cv2.INTER_CUBIC)
    if margen > 0:
        m = int(img.shape[0] * margen)
        img = cv2.copyMakeBorder(img, m, m, m, m, cv2.BORDER_REPLICATE)
    return img


def main():
    ap = argparse.ArgumentParser(description="Extrae puntos de LSC70ANH con MediaPipe")
    ap.add_argument("--datos", default=str(RAIZ / "LSC70" / "LSC70ANH"))
    ap.add_argument("--salida", default=str(RAIZ / "datos" / "puntos_lsc70.csv"))
    ap.add_argument("--escala", type=float, default=2.0)
    ap.add_argument("--margen", type=float, default=0.25)
    ap.add_argument("--muestra", type=int, default=0,
                    help="procesar solo N fotos al azar (para probar)")
    args = ap.parse_args()

    mp = importar_mediapipe()
    raiz_datos = Path(args.datos)
    if not raiz_datos.is_dir():
        raise SystemExit(f"No encuentro la carpeta del dataset: {raiz_datos}")

    # Lista de (voluntario, letra, fotograma, ruta)
    fotos = []
    for carpeta_vol in sorted(raiz_datos.iterdir()):
        for letra in LETRAS:
            for ruta in sorted((carpeta_vol / letra).glob("*.jpg")):
                fotograma = int(ruta.stem.rsplit("_", 1)[1])
                fotos.append((carpeta_vol.name, letra, fotograma, ruta))
    if args.muestra:
        random.seed(0)
        fotos = random.sample(fotos, min(args.muestra, len(fotos)))
    print(f"Fotos a procesar: {len(fotos)}")

    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)

    detectadas, fallidas = Counter(), Counter()
    inicio = time.time()
    # static_image_mode=True: cada foto se analiza por separado (no es video).
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1,
                                  model_complexity=1,
                                  min_detection_confidence=0.3) as manos, \
            open(salida, "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(["voluntario", "letra", "fotograma", "mano"] + COLUMNAS_PUNTOS)
        for i, (vol, letra, fotograma, ruta) in enumerate(fotos, 1):
            # cv2.imread no abre rutas con tildes en Windows: leemos los bytes
            # con Python y los decodificamos con OpenCV.
            img = cv2.imdecode(np.frombuffer(ruta.read_bytes(), np.uint8),
                               cv2.IMREAD_COLOR)
            if img is None:
                fallidas[letra] += 1
                continue
            img = preparar_imagen(img, args.escala, args.margen)
            res = manos.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            vector = None
            if res.multi_hand_landmarks:
                alto, ancho = img.shape[:2]
                vector = normalizar_puntos(
                    landmarks_a_pixeles(res.multi_hand_landmarks[0], ancho, alto))
            if vector is None:
                fallidas[letra] += 1
                continue
            mano = res.multi_handedness[0].classification[0].label
            escritor.writerow([vol, letra, fotograma, mano] + [f"{v:.5f}" for v in vector])
            detectadas[letra] += 1
            if i % 1000 == 0:
                print(f"  {i}/{len(fotos)} fotos ({time.time() - inicio:.0f} s)")

    # Reporte: cuántas fotos por letra se detectaron y cuántas fallaron.
    print("\nLetra  detectadas  fallidas  %detección")
    for letra in LETRAS:
        d, fa = detectadas[letra], fallidas[letra]
        if d + fa:
            print(f"{letra:<6} {d:>10} {fa:>9} {100 * d / (d + fa):>10.1f}")
    td, tf = sum(detectadas.values()), sum(fallidas.values())
    print(f"TOTAL  {td:>10} {tf:>9} {100 * td / max(1, td + tf):>10.1f}")
    print(f"\nGuardado en {salida} ({time.time() - inicio:.0f} s)")


if __name__ == "__main__":
    main()
