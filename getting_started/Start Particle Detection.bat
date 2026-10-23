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
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $c = New-Object System.Net.Sockets.TcpClient; $c.Connect('127.0.0.1', 8502); $c.Close(); exit 0 } catch { exit 1 }"
if not errorlevel 1 (
  start "" "%URL%"
  exit /b 0
)

start /b "" powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command "$deadline = (Get-Date).AddSeconds(30); while ((Get-Date) -lt $deadline) { try { $c = New-Object System.Net.Sockets.TcpClient; $c.Connect('127.0.0.1', 8502); $c.Close(); Start-Process 'http://127.0.0.1:8502'; exit 0 } catch { Start-Sleep -Milliseconds 500 } }; Add-Content -LiteralPath (Join-Path $env:USERPROFILE 'AppData\Local\ParticleDetection\launcher.log') -Value ((Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + ' Streamlit did not accept connections on 127.0.0.1:8502 within 30 seconds')"

>>"%LOG%" echo %DATE% %TIME% starting Streamlit on %URL%
"%STREAMLIT%" run app.py --server.headless true --server.address 127.0.0.1 --server.port 8502 >> "%LOG%" 2>&1
exit /b %ERRORLEVEL%
