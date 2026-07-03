@echo off
REM ============================================================
REM EGX daily levels email — Windows Task Scheduler entrypoint.
REM Schedule "every day at 08:30"; the script self-skips Fri/Sat.
REM
REM Email setup (one time): run configure_smtp.bat after putting
REM your Gmail App Password in it. Until then this still writes
REM briefings\levels_YYYY-MM-DD.html but won't email.
REM ============================================================

cd /d "D:\EGX MCP\egx-mcp\egx-mcp"

if not exist "logs" mkdir "logs"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set "stamp=%%i"
if "%stamp%"=="" set "stamp=%date:~10,4%-%date:~4,2%-%date:~7,2%"

python -m scripts.levels_email >> "logs\levels_%stamp%.log" 2>&1
exit /b %ERRORLEVEL%
