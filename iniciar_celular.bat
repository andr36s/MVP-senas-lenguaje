@echo off
rem ======================================================================
rem  Senas LSC - probar desde el celular (misma red Wi-Fi)
rem  Sirve la app por HTTPS en la red local. Ver servidor_celular.py.
rem  OJO: cualquier dispositivo de la misma Wi-Fi puede abrir la pagina.
rem  Usalo solo en redes de confianza y cierralo al terminar.
rem ======================================================================
title Senas LSC - servidor para celular
for %%I in ("%~dp0.") do set "RAIZ=%%~sI"
cd /d "%RAIZ%"

if not exist "venv\Scripts\activate.bat" (
  echo No encuentro el entorno virtual. Revisa la seccion de instalacion del README.
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
python servidor_celular.py
pause
