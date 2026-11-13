"""
Etapa 3: grabar ejemplos propios con la webcam.

Por qué: el dataset tiene 70 voluntarios grabados con celular. Agregar
ejemplos de quien va a usar la app (con su cámara, su luz y su mano) ayuda a
que el modelo funcione mejor en la demostración real.

Para qué: guarda en datos/mis_datos.csv los 63 valores normalizados de la
mano, con el mismo formato que datos/puntos_lsc70.csv, para que entrenar.py
los mezcle con el dataset. NO se guarda ninguna imagen, solo números.

Uso:
    python grabar_mis_datos.py --nombre andres

Controles (con la ventana de la cámara seleccionada):
    Tecla de una letra (a-z): guarda la mano actual como esa letra.
                              Mantenerla presionada graba varias muestras.
    Tecla 1: guarda la mano como Ñ (no todas las teclas ñ llegan a OpenCV).
    Tecla Esc: salir.
"""
import argparse
import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2

from comun import (COLUMNAS_PUNTOS, importar_mediapipe, landmarks_a_pixeles,
                   normalizar_puntos)

RAIZ = Path(__file__).resolve().parent

CONSENTIMIENTO = """
=====================================================================
 AUTORIZACIÓN DE TRATAMIENTO DE DATOS (Ley 1581 de 2012, Colombia)
=====================================================================
 Este programa usará la cámara para calcular la posición de 21 puntos
 de tu mano. Solo se guardan esos números (no fotos ni video) en
 datos/mis_datos.csv, junto con el nombre o seudónimo que escribiste y
 la fecha. Finalidad: entrenar el modelo de reconocimiento de letras
 de este proyecto académico. Puedes pedir que se borren en cualquier
 momento eliminando tus filas del archivo. Ese archivo no se sube al
 repositorio (está en .gitignore).
=====================================================================
"""


def main():
    ap = argparse.ArgumentParser(description="Graba ejemplos propios de letras LSC")
    ap.add_argument("--nombre", required=True,
                    help="nombre o seudónimo de quien graba (ej. andres)")
    ap.add_argument("--salida", default=str(RAIZ / "datos" / "mis_datos.csv"))
    ap.add_argument("--camara", type=int, default=0)
    args = ap.parse_args()

    print(CONSENTIMIENTO)
    if input('Escribe "acepto" para continuar: ').strip().lower() != "acepto":
        raise SystemExit("Sin autorización no se graba nada.")

    mp = importar_mediapipe()
    dibujo = mp.solutions.drawing_utils
    voluntario = f"propio_{args.nombre.strip().lower()}"

    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not salida.exists()
    conteo = Counter()

    # CAP_DSHOW abre la webcam mucho más rápido en Windows.
    camara = cv2.VideoCapture(args.camara, cv2.CAP_DSHOW)
    if not camara.isOpened():
        raise SystemExit("No pude abrir la cámara. ¿La está usando otro programa?")

    # static_image_mode=False: modo video, sigue la mano entre cuadros (más rápido).
    with mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                  model_complexity=1,
                                  min_detection_confidence=0.6,
                                  min_tracking_confidence=0.5) as manos, \
            open(salida, "a", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        if nuevo:
            escritor.writerow(["voluntario", "letra", "fotograma", "mano"] + COLUMNAS_PUNTOS)
        mensaje = "Haz una letra y presiona su tecla"
        while True:
            ok, cuadro = camara.read()
            if not ok:
                break
            res = manos.process(cv2.cvtColor(cuadro, cv2.COLOR_BGR2RGB))
            vector = None
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                alto, ancho = cuadro.shape[:2]
                vector = normalizar_puntos(landmarks_a_pixeles(lm, ancho, alto))
                dibujo.draw_landmarks(cuadro, lm, mp.solutions.hands.HAND_CONNECTIONS)

            # Mostramos la imagen en espejo, como un espejo real (más natural).
            vista = cv2.flip(cuadro, 1)
            cv2.putText(vista, mensaje, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0) if vector else (0, 0, 255), 2)
            resumen = " ".join(f"{k}:{v}" for k, v in sorted(conteo.items()))
            cv2.putText(vista, resumen[:80], (10, vista.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("Grabar mis datos - Esc para salir", vista)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == 27:  # Esc
                break
            letra = None
            if ord("a") <= tecla <= ord("z"):
                letra = chr(tecla).upper()
            elif tecla in (ord("1"), 241, 209):  # 1, ñ, Ñ
                letra = "NN"
            if letra:
                if vector is None:
                    mensaje = "No veo la mano: no se guardó"
                    continue
                mano = res.multi_handedness[0].classification[0].label
                escritor.writerow([voluntario, letra, conteo[letra], mano]
                                  + [f"{v:.5f}" for v in vector])
                f.flush()
                conteo[letra] += 1
                nombre = "Ñ" if letra == "NN" else letra
                mensaje = f"Guardada {nombre} ({conteo[letra]})"

    camara.release()
    cv2.destroyAllWindows()
    print(f"\n{sum(conteo.values())} muestras guardadas en {salida} "
          f"({datetime.now():%Y-%m-%d %H:%M}).")
    for letra, n in sorted(conteo.items()):
        print(f"  {letra}: {n}")


if __name__ == "__main__":
    main()
