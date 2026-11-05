@echo off
setlocal EnableExtensions
cd /d "%~dp0.."

set "LOGDIR=%USERPROFILE%\AppData\Local\ParticleDetection"
set "LOG=%LOGDIR%\launcher.log"
set "STREAMLIT=%CD%\.venv\Scripts\streamlit.exe"
set "URL=http://127.0.0.1:8502"

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

if not exist "%STREAMLIT%" (
  >>"%LOG%" echo %DATE% %TIME% missing .venv\Scripts\streamlit.exe at "%STREAMLIT%"
  echo Particle Detection could not start.
  echo Missing "%STREAMLIT%"
  echo.
  echo From this folder, run:
  echo   py -3 -m venv .venv
  echo   .venv\Scripts\activate
  echo   python -m pip install --upgrade pip
  echo   pip install -r getting_started\requirements.txt
  echo   python -m pip install --no-deps "Printrun==2.2.0"
  echo.
  echo Log: "%LOG%"
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.MessageBox]::Show('The app environment is missing. From the project folder run: py -3 -m venv .venv, then pip install -r getting_started\requirements.txt and python -m pip install --no-deps Printrun==2.2.0. Log: %LOG%', 'Particle Detection')"
  pause
  exit /b 1
)

echo Particle Detection
echo Leave this window open. Closing it stops the app.
echo Log: "%LOG%"

powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $c = New-Object System.Net.Sockets.TcpClient; $c.Connect('127.0.0.1', 8502); $c.Close(); exit 0 } catch { exit 1 }"
if not errorlevel 1 (
  start "" "%URL%"
  exit /b 0
)

start /b "" powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command "$deadline = (Get-Date).AddSeconds(30); while ((Get-Date) -lt $deadline) { try { $c = New-Object System.Net.Sockets.TcpClient; $c.Connect('127.0.0.1', 8502); $c.Close(); Start-Process 'http://127.0.0.1:8502'; exit 0 } catch { Start-Sleep -Milliseconds 500 } }; Add-Content -LiteralPath (Join-Path $env:USERPROFILE 'AppData\Local\ParticleDetection\launcher.log') -Value ((Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + ' Streamlit did not accept connections on 127.0.0.1:8502 within 30 seconds')"

>>"%LOG%" echo %DATE% %TIME% starting Streamlit on %URL%
"%STREAMLIT%" run app.py --server.headless true --server.address 127.0.0.1 --server.port 8502 >> "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo.
  echo Streamlit stopped with exit code %RC%.
  echo See "%LOG%"
  pause
)
exit /b %RC%
