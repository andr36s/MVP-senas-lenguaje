"""
Funciones compartidas por todos los scripts del proyecto "Señas".

Por qué existe este archivo: la normalización de los puntos de la mano tiene
que ser EXACTAMENTE la misma al extraer el dataset, al grabar datos propios y
al predecir. Si cada script la implementara por su cuenta, una pequeña
diferencia haría que el modelo reciba en uso real datos distintos a los que
vio al entrenar. (La página web la replica en JavaScript: ver index.html.)
"""
import os
import sys

# Número de puntos que entrega MediaPipe Hands por mano, y valores por punto.
N_PUNTOS = 21
N_VALORES = N_PUNTOS * 3  # x, y, z -> 63 números

# Índices de MediaPipe que usamos para normalizar.
MUNECA = 0
BASE_DEDO_MEDIO = 9  # nudillo (MCP) del dedo medio

COLUMNAS_PUNTOS = [f"{eje}{i}" for i in range(N_PUNTOS) for eje in ("x", "y", "z")]


def _ruta_corta(ruta):
    """Devuelve la ruta corta 8.3 de Windows (sin tildes ni ñ), si existe."""
    import ctypes
    buf = ctypes.create_unicode_buffer(1024)
    n = ctypes.windll.kernel32.GetShortPathNameW(ruta, buf, len(buf))
    return buf.value if 0 < n < len(buf) else ruta


def preparar_rutas_windows():
    """
    MediaPipe (compilado en C++) no encuentra sus archivos de modelo si la
    ruta de instalación tiene caracteres no-ASCII (ej. C:\\Users\\Andrés\\...).
    Solución: antes de importar mediapipe, cambiamos las rutas de búsqueda de
    Python por su versión corta de Windows (C:\\Users\\ANDRS~1\\...), que
    apunta a los mismos archivos pero sin tildes. Debe llamarse ANTES de
    `import mediapipe`.
    """
    if os.name != "nt":
        return
    sys.path[:] = [
        _ruta_corta(p) if p and not p.isascii() and os.path.exists(p) else p
        for p in sys.path
    ]


def importar_mediapipe():
    """Importa mediapipe aplicando antes el arreglo de rutas de Windows."""
    preparar_rutas_windows()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # menos mensajes en consola
    import mediapipe as mp
    return mp


def normalizar_puntos(puntos_px):
    """
    Convierte 21 puntos (x, y, z) en píxeles en 63 números invariantes a la
    posición y al tamaño de la mano en la imagen.

    1. Restamos la muñeca: así da igual en qué parte de la imagen esté la mano.
    2. Dividimos por la distancia muñeca -> nudillo del dedo medio (tamaño de
       la palma): así da igual si la mano está cerca o lejos de la cámara.
       Usamos la palma y no la mano completa porque la palma mide lo mismo
       con el puño cerrado o abierto; la mano completa no.

    puntos_px: lista de 21 tuplas (x, y, z) en píxeles.
    Devuelve: lista de 63 floats, o None si la mano es degenerada.
    """
    import numpy as np
    p = np.asarray(puntos_px, dtype=np.float32).reshape(N_PUNTOS, 3)
    p = p - p[MUNECA]
    tamano = float(np.linalg.norm(p[BASE_DEDO_MEDIO]))
    if tamano < 1e-6:
        return None
    return (p / tamano).reshape(-1).tolist()


def landmarks_a_pixeles(landmarks, ancho, alto):
    """
    MediaPipe da x en fracción del ancho, y en fracción del alto, y z en
    aproximadamente la misma escala que x. Pasamos todo a píxeles para que la
    forma de la mano no se deforme cuando la imagen no es cuadrada (la webcam
    es 4:3, las fotos del dataset son 1:1).
    """
    return [(lm.x * ancho, lm.y * alto, lm.z * ancho) for lm in landmarks.landmark]


def espejar(vector63):
    """
    Refleja la mano horizontalmente (mano derecha <-> izquierda) cambiando el
    signo de las x. Como ya están relativas a la muñeca, basta con negarlas.
    Sirve para que el modelo reconozca la letra hecha con cualquiera de las
    dos manos (personas zurdas, o la cámara en modo espejo).
    """
    v = list(vector63)
    for i in range(0, len(v), 3):
        v[i] = -v[i]
    return v
