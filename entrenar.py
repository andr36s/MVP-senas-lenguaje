"""
Etapa 4: entrenar la red neuronal con Keras 3.

Por qué una red Dense pequeña: la entrada son solo 63 números (no una
imagen), así que no hace falta una red convolucional. Una red de dos capas
ocultas entrena en segundos en CPU y es fácil de explicar.

Para qué: produce modelo_senas.keras y etiquetas.json, que usa api.py, y
deja en resultados/ las métricas y la matriz de confusión para evaluar si el
modelo es confiable.

Uso:
    python entrenar.py
"""
import json
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import matplotlib
import numpy as np
import pandas as pd
from keras import layers
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

from comun import COLUMNAS_PUNTOS, espejar

matplotlib.use("Agg")  # dibujar sin abrir ventanas
import matplotlib.pyplot as plt  # noqa: E402

RAIZ = Path(__file__).resolve().parent
DATOS_LSC70 = RAIZ / "datos" / "puntos_lsc70.csv"
DATOS_PROPIOS = RAIZ / "datos" / "mis_datos.csv"
RESULTADOS = RAIZ / "resultados"
SEMILLA = 42

# Letras que se hacen con movimiento en LSC. Un modelo que mira una sola
# pose no las puede distinguir bien, así que quedan fuera del MVP.
# Se eligieron midiendo el movimiento entre los 6 fotogramas de cada seña
# (ver la tabla que imprime medir_movimiento) y revisando las fotos:
# las 21 letras estáticas se mueven menos de 0,13 tamaños de palma, y estas
# 6 más de 0,38. Hay un corte claro entre los dos grupos.
LETRAS_CON_MOVIMIENTO = ["G", "H", "J", "NN", "S", "Z"]

UMBRAL_CONFIANZA = 0.70  # por debajo, la API responde "no estoy seguro"


def nombre_letra(letra):
    return "Ñ" if letra == "NN" else letra


def medir_movimiento(df):
    """
    Para cada voluntario y letra, mide cuánto cambian los 21 puntos entre
    los 6 fotogramas (distancia media de cada fotograma a la pose promedio,
    en unidades de "tamaño de palma"). Una letra estática da un valor bajo;
    una con movimiento, alto. Devuelve la mediana por letra.
    """
    filas = []
    for (vol, letra), g in df.groupby(["voluntario", "letra"]):
        if len(g) < 4:
            continue
        p = g[COLUMNAS_PUNTOS].to_numpy().reshape(len(g), 21, 3)
        media = p.mean(axis=0)
        filas.append((letra, float(np.linalg.norm(p - media, axis=2).mean())))
    mov = pd.DataFrame(filas, columns=["letra", "movimiento"])
    return mov.groupby("letra")["movimiento"].median().sort_values()


def dividir_por_voluntario(df):
    """
    Divide 70 % / 15 % / 15 % POR VOLUNTARIO: todas las fotos de una persona
    quedan en un solo grupo. Si una persona estuviera en entrenamiento y en
    prueba, el modelo podría "reconocer a la persona" en vez de la letra y la
    exactitud de prueba saldría inflada (fuga de datos).
    Los datos propios (voluntario "propio_...") van solo a entrenamiento.
    """
    propios = df["voluntario"].str.startswith("propio_")
    lsc = df[~propios]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEMILLA)
    i_ent, i_resto = next(gss.split(lsc, groups=lsc["voluntario"]))
    resto = lsc.iloc[i_resto]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEMILLA)
    i_val, i_pru = next(gss2.split(resto, groups=resto["voluntario"]))
    ent = pd.concat([lsc.iloc[i_ent], df[propios]])
    return ent, resto.iloc[i_val], resto.iloc[i_pru]


def construir_modelo(n_letras):
    """Input(63) -> Dense -> Dropout -> Dense -> Dense(n_letras, softmax)."""
    modelo = keras.Sequential([
        keras.Input(shape=(63,)),
        layers.Dense(128, activation="relu"),
        # Dropout apaga al azar el 30 % de las neuronas en cada paso de
        # entrenamiento: obliga a la red a no depender de unas pocas y reduce
        # el sobreajuste.
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(n_letras, activation="softmax"),
    ], name="red_senas")
    modelo.compile(optimizer=keras.optimizers.Adam(1e-3),
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
    return modelo


def aumentar_con_espejo(x, y):
    """Duplica los datos con la mano reflejada: sirve para diestros y zurdos."""
    x_esp = np.array([espejar(v) for v in x], dtype=np.float32)
    return np.concatenate([x, x_esp]), np.concatenate([y, y])


def guardar_matriz_confusion(y_real, y_pred, letras, ruta):
    mc = confusion_matrix(y_real, y_pred, labels=range(len(letras)))
    # Normalizada por fila: cada fila suma 100 % (qué se predijo para cada letra real).
    mc_pct = mc / np.maximum(mc.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(mc_pct, cmap="Blues", vmin=0, vmax=1)
    nombres = [nombre_letra(l) for l in letras]
    ax.set_xticks(range(len(letras)), nombres)
    ax.set_yticks(range(len(letras)), nombres)
    ax.set_xlabel("Letra predicha")
    ax.set_ylabel("Letra real")
    ax.set_title("Matriz de confusión (conjunto de prueba, % por fila)")
    for i in range(len(letras)):
        for j in range(len(letras)):
            if mc_pct[i, j] >= 0.05:
                ax.text(j, i, f"{100 * mc_pct[i, j]:.0f}", ha="center", va="center",
                        fontsize=7, color="white" if mc_pct[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def main():
    keras.utils.set_random_seed(SEMILLA)
    RESULTADOS.mkdir(exist_ok=True)

    df = pd.read_csv(DATOS_LSC70)
    print(f"Dataset LSC70: {len(df)} muestras de {df['voluntario'].nunique()} voluntarios")
    if DATOS_PROPIOS.exists():
        propios = pd.read_csv(DATOS_PROPIOS)
        print(f"Datos propios: {len(propios)} muestras")
        df = pd.concat([df, propios], ignore_index=True)

    print("\nMovimiento entre fotogramas (mediana, en tamaños de palma):")
    movimiento = medir_movimiento(df[~df["voluntario"].str.startswith("propio_")])
    for letra, valor in movimiento.items():
        marca = "  <- con movimiento, excluida" if letra in LETRAS_CON_MOVIMIENTO else ""
        print(f"  {nombre_letra(letra):<3} {valor:.3f}{marca}")

    df = df[~df["letra"].isin(LETRAS_CON_MOVIMIENTO)]
    letras = sorted(df["letra"].unique())
    indice = {l: i for i, l in enumerate(letras)}

    ent, val, pru = dividir_por_voluntario(df)
    for nombre, parte in (("entrenamiento", ent), ("validación", val), ("prueba", pru)):
        print(f"{nombre:>14}: {len(parte):>6} muestras, {parte['voluntario'].nunique()} voluntarios")
    assert not (set(ent.voluntario) & set(pru.voluntario)), "fuga de datos"
    assert not (set(ent.voluntario) & set(val.voluntario)), "fuga de datos"
    assert not (set(val.voluntario) & set(pru.voluntario)), "fuga de datos"

    def a_xy(parte):
        return (parte[COLUMNAS_PUNTOS].to_numpy(np.float32),
                parte["letra"].map(indice).to_numpy())

    x_ent, y_ent = aumentar_con_espejo(*a_xy(ent))
    x_val, y_val = aumentar_con_espejo(*a_xy(val))
    x_pru, y_pru = a_xy(pru)

    modelo = construir_modelo(len(letras))
    modelo.summary()

    ruta_modelo = RAIZ / "modelo_senas.keras"
    callbacks = [
        # Detiene el entrenamiento cuando la pérdida en validación deja de
        # mejorar por 15 épocas, y recupera los mejores pesos: evita sobreajuste.
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                      restore_best_weights=True),
        # Guarda en disco el mejor modelo visto hasta ahora (según validación).
        keras.callbacks.ModelCheckpoint(ruta_modelo, monitor="val_loss",
                                        save_best_only=True),
    ]
    historia = modelo.fit(x_ent, y_ent, validation_data=(x_val, y_val),
                          epochs=300, batch_size=64, callbacks=callbacks, verbose=2)

    # Evaluación final con el conjunto de prueba, que el modelo nunca vio.
    modelo = keras.saving.load_model(ruta_modelo)
    prob = modelo.predict(x_pru, verbose=0)
    y_pred = prob.argmax(axis=1)
    exactitud = accuracy_score(y_pru, y_pred)
    seguras = prob.max(axis=1) >= UMBRAL_CONFIANZA
    exactitud_seguras = accuracy_score(y_pru[seguras], y_pred[seguras])
    prob_esp = modelo.predict(np.array([espejar(v) for v in x_pru], np.float32), verbose=0)
    exactitud_espejo = accuracy_score(y_pru, prob_esp.argmax(axis=1))

    nombres = [nombre_letra(l) for l in letras]
    reporte = classification_report(y_pru, y_pred, labels=range(len(letras)),
                                     target_names=nombres, digits=3, zero_division=0)
    print(f"\nExactitud en prueba: {100 * exactitud:.1f} %")
    print(f"Exactitud en prueba con la mano reflejada (zurdos): {100 * exactitud_espejo:.1f} %")
    print(f"Con umbral de {UMBRAL_CONFIANZA:.0%}: responde en el {100 * seguras.mean():.1f} % "
          f"de los casos y acierta el {100 * exactitud_seguras:.1f} % de esas veces")
    print("\nExactitud por letra (recall = de las veces que era esa letra, cuántas acertó):")
    print(reporte)

    # Exactitud por persona: si varía mucho, el modelo funciona mejor para
    # unas manos que para otras (posible sesgo).
    por_voluntario = (pd.Series(y_pred == y_pru, index=pru["voluntario"].to_numpy())
                      .groupby(level=0).mean().sort_values())
    print("Exactitud por voluntario de prueba: "
          + ", ".join(f"{v} {100 * a:.0f} %" for v, a in por_voluntario.items()))

    guardar_matriz_confusion(y_pru, y_pred, letras, RESULTADOS / "matriz_confusion.png")

    # Curvas de aprendizaje: muestran si hubo sobreajuste.
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for clave, eje, titulo in (("loss", ax[0], "Pérdida"), ("accuracy", ax[1], "Exactitud")):
        eje.plot(historia.history[clave], label="entrenamiento")
        eje.plot(historia.history[f"val_{clave}"], label="validación")
        eje.set_title(titulo)
        eje.set_xlabel("época")
        eje.legend()
    fig.tight_layout()
    fig.savefig(RESULTADOS / "curvas_entrenamiento.png", dpi=110)
    plt.close(fig)

    por_letra = {nombres[i]: float((y_pred[y_pru == i] == i).mean())
                 for i in range(len(letras)) if (y_pru == i).any()}
    metricas = {
        "exactitud_prueba": exactitud,
        "exactitud_prueba_espejo": exactitud_espejo,
        "umbral_confianza": UMBRAL_CONFIANZA,
        "cobertura_con_umbral": float(seguras.mean()),
        "exactitud_con_umbral": exactitud_seguras,
        "exactitud_por_letra": por_letra,
        "exactitud_por_voluntario_prueba": {k: float(v) for k, v in por_voluntario.items()},
        "movimiento_por_letra": {nombre_letra(k): v for k, v in movimiento.items()},
        "voluntarios": {"entrenamiento": sorted(ent.voluntario.unique()),
                        "validacion": sorted(val.voluntario.unique()),
                        "prueba": sorted(pru.voluntario.unique())},
        "epocas": len(historia.history["loss"]),
    }
    (RESULTADOS / "metricas.json").write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")
    (RESULTADOS / "reporte_por_letra.txt").write_text(reporte, encoding="utf-8")

    etiquetas = {"letras": letras, "nombres": nombres,
                 "umbral_confianza": UMBRAL_CONFIANZA}
    (RAIZ / "etiquetas.json").write_text(
        json.dumps(etiquetas, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardados: {ruta_modelo.name}, etiquetas.json y la carpeta resultados/")


if __name__ == "__main__":
    main()
