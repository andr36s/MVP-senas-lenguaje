"""
Arranca la app para probarla desde un celular en la misma red Wi-Fi.

Por qué HTTPS: los navegadores de celular solo dejan usar la cámara en
páginas seguras (HTTPS). En el computador funciona con http://127.0.0.1
porque "localhost" es una excepción, pero desde el celular la dirección es
la IP del computador (ej. 192.168.1.7) y ahí sí exige HTTPS.

Para qué: genera un certificado autofirmado para la IP del computador en la
red local y sirve la misma app de api.py en https://<IP>:8443. El celular
mostrará una advertencia porque el certificado no lo firmó una autoridad
reconocida: es esperado, se acepta una vez ("Avanzado" -> "Continuar").

Diferencia de privacidad con iniciar_windows.bat: aquí el servidor escucha
en TODA la red local, no solo en 127.0.0.1. Cualquier dispositivo de la
misma Wi-Fi podría abrir la página. Úsalo solo en redes de confianza y
ciérralo al terminar. El video sigue sin salir del celular: al computador
solo llegan los 63 números de cada consulta.

Uso:
    python servidor_celular.py
"""
import shutil
import socket
import subprocess
from pathlib import Path

import uvicorn

RAIZ = Path(__file__).resolve().parent
CARPETA = RAIZ / "certificados"
PUERTO = 8443

# openssl viene incluido con Git para Windows.
OPENSSL_POSIBLES = [
    r"C:\Program Files\Git\usr\bin\openssl.exe",
    r"C:\Program Files\Git\mingw64\bin\openssl.exe",
]


def ip_local():
    """
    IP del computador en la red local. Abrir un socket UDP hacia una IP
    externa no envía ningún paquete: solo le pregunta al sistema operativo
    qué interfaz de red usaría para salir a internet (la de la Wi-Fi).
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]


def buscar_openssl():
    encontrado = shutil.which("openssl")
    if encontrado:
        return encontrado
    for ruta in OPENSSL_POSIBLES:
        if Path(ruta).exists():
            return ruta
    raise SystemExit("No encuentro openssl. Viene con Git para Windows: "
                     "https://git-scm.com/download/win")


def crear_certificado(ip):
    """Crea (o reutiliza) un certificado autofirmado válido para esta IP."""
    CARPETA.mkdir(exist_ok=True)
    cert, llave, marca = CARPETA / "cert.pem", CARPETA / "llave.pem", CARPETA / "ip.txt"
    if cert.exists() and llave.exists() and marca.exists() and marca.read_text() == ip:
        return cert, llave
    print(f"Creando certificado HTTPS para {ip} ...")
    subprocess.run([
        buscar_openssl(), "req", "-x509", "-newkey", "rsa:2048", "-nodes",
        "-keyout", str(llave), "-out", str(cert), "-days", "365",
        "-subj", "/CN=Senas LSC (prueba local)",
        "-addext", f"subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost",
    ], check=True, capture_output=True)
    marca.write_text(ip)
    return cert, llave


def main():
    ip = ip_local()
    cert, llave = crear_certificado(ip)
    url = f"https://{ip}:{PUERTO}"
    print("\n" + "=" * 60)
    print("  Abre esta dirección en el navegador del celular")
    print("  (conectado a la MISMA red Wi-Fi que este computador):\n")
    print(f"      {url}\n")
    print("  1. Verás una advertencia de seguridad: toca 'Avanzado' o")
    print("     'Mostrar detalles' y luego 'Continuar' o 'Visitar sitio'.")
    print("  2. Permite el acceso a la cámara.")
    print("  Para detener el servidor: Ctrl+C o cierra esta ventana.")
    print("=" * 60 + "\n")
    # host 0.0.0.0 = escuchar en todas las interfaces de red (no solo 127.0.0.1)
    uvicorn.run("api:app", host="0.0.0.0", port=PUERTO,
                ssl_certfile=str(cert), ssl_keyfile=str(llave), app_dir=str(RAIZ))


if __name__ == "__main__":
    main()
