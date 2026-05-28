@echo off
REM ============================================================
REM Daily EGX Briefing — Windows Task Scheduler entrypoint
REM Runs at 09:00 Cairo time on EGX trading days (Sun-Thu).
REM
REM Setup:
REM   1. Edit configure_smtp.bat with your Gmail App Password
REM   2. Run configure_smtp.bat ONCE to set persistent env vars
REM   3. In Task Scheduler, point the action to THIS file
REM
REM The script itself checks the day-of-week, so scheduling
REM "every day at 09:00" is fine — it self-skips Fri/Sat.
REM ============================================================

cd /d "D:\EGX MCP\egx-mcp\egx-mcp"

REM Ensure log directory exists
if not exist "logs" mkdir "logs"

REM Use PowerShell to get a portable yyyy-MM-dd timestamp.
REM (wmic was removed on Windows 11 24H2+; PowerShell ships everywhere.)
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set "stamp=%%i"

REM Fallback if PowerShell isn't available — use the locale-dependent %date%
if "%stamp%"=="" set "stamp=%date:~10,4%-%date:~4,2%-%date:~7,2%"

REM Run the briefing — log everything for audit
python -m scripts.daily_briefing >> "logs\briefing_%stamp%.log" 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo Briefing FAILED with exit code %ERRORLEVEL% >> "logs\briefing_%stamp%.log"
    exit /b %ERRORLEVEL%
)

REM Accumulate the evidence base (Gate #1): grade today's briefing and regrade
REM prior ones as realized prices come in. Non-fatal — never fail the briefing.
python -m tests.grade_briefings >> "logs\briefing_%stamp%.log" 2>&1

exit /b 0
