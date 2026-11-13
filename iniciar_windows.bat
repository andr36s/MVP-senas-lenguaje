@echo off
rem ======================================================================
rem  Senas LSC - arranque con doble clic
rem  1) Se ubica en la carpeta del proyecto usando la ruta corta de Windows
rem     (sin tildes, para que MediaPipe y TensorFlow no fallen).
rem  2) Activa el entorno virtual.
rem  3) Abre el navegador y arranca el servidor solo en 127.0.0.1
rem     (nadie mas en la red puede conectarse).
rem ======================================================================
title Senas LSC - servidor
for %%I in ("%~dp0.") do set "RAIZ=%%~sI"
cd /d "%RAIZ%"

if not exist "venv\Scripts\activate.bat" (
  echo No encuentro el entorno virtual. Crealo primero:
  echo    py -3.12 -m venv venv
  echo    venv\Scripts\pip install -r requirements.txt
  echo    venv\Scripts\pip install --no-deps mediapipe==0.10.21
  pause
  exit /b 1
)
if not exist "modelo_senas.keras" (
  echo No encuentro modelo_senas.keras. Entrena primero con:
  echo    venv\Scripts\python entrenar.py
  pause
  exit /b 1
)

call "venv\Scripts\activate.bat"

echo Iniciando el servidor en http://127.0.0.1:8000 ...
echo Para detenerlo, cierra esta ventana o presiona Ctrl+C.
rem Abre el navegador unos segundos despues, cuando el modelo ya cargo.
start "" /min cmd /c "timeout /t 8 /nobreak >nul & start http://127.0.0.1:8000"
python -m uvicorn api:app --host 127.0.0.1 --port 8000
pause
