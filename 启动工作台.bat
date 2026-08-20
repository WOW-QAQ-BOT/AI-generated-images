@echo off
chcp 65001 >nul
setlocal
pushd "%~dp0"

set "PYTHON_EXE="

:TRY_PYTHON
set "CANDIDATE_PYTHON="
for /f "tokens=1,* delims=#" %%A in (`python -c "import tkinter; import sys; sys.exit(3) if sys.version_info < (3, 10) else None; sys.stdout.reconfigure(encoding='utf-8', errors='strict'); print('CODEX_PYTHON_OK#' + sys.executable)" 2^>nul`) do (
    if "%%A"=="CODEX_PYTHON_OK" (
        set "CANDIDATE_PYTHON=%%B"
    )
)
if defined CANDIDATE_PYTHON if exist "%CANDIDATE_PYTHON%" (
    set "PYTHON_EXE=%CANDIDATE_PYTHON%"
    goto :PYTHON_FOUND
)

:TRY_PY_LAUNCHER
set "CANDIDATE_PYTHON="
for /f "tokens=1,* delims=#" %%A in (`py -3 -c "import tkinter; import sys; sys.exit(3) if sys.version_info < (3, 10) else None; sys.stdout.reconfigure(encoding='utf-8', errors='strict'); print('CODEX_PYTHON_OK#' + sys.executable)" 2^>nul`) do (
    if "%%A"=="CODEX_PYTHON_OK" (
        set "CANDIDATE_PYTHON=%%B"
    )
)
if defined CANDIDATE_PYTHON if exist "%CANDIDATE_PYTHON%" (
    set "PYTHON_EXE=%CANDIDATE_PYTHON%"
    goto :PYTHON_FOUND
)

:TRY_PYTHON3
set "CANDIDATE_PYTHON="
for /f "tokens=1,* delims=#" %%A in (`python3 -c "import tkinter; import sys; sys.exit(3) if sys.version_info < (3, 10) else None; sys.stdout.reconfigure(encoding='utf-8', errors='strict'); print('CODEX_PYTHON_OK#' + sys.executable)" 2^>nul`) do (
    if "%%A"=="CODEX_PYTHON_OK" (
        set "CANDIDATE_PYTHON=%%B"
    )
)
if defined CANDIDATE_PYTHON if exist "%CANDIDATE_PYTHON%" (
    set "PYTHON_EXE=%CANDIDATE_PYTHON%"
    goto :PYTHON_FOUND
)

:NO_PYTHON
echo 未找到可用的 Python 3.10+ 或 Tkinter。建议安装带 Tkinter 的 Python 3.10 或 3.11，然后重新运行此文件。
pause
popd
endlocal & exit /b 1

:PYTHON_FOUND
if not exist "launcher.pyw" goto :MISSING_LAUNCHER
for %%I in ("%PYTHON_EXE%") do set "PYTHON_DIR=%%~dpI"

if exist "%PYTHON_DIR%pythonw.exe" (
    start "" /b "%PYTHON_DIR%pythonw.exe" "%CD%\launcher.pyw"
) else (
    start "" /b "%PYTHON_EXE%" "%CD%\launcher.pyw"
)
if errorlevel 1 goto :LAUNCH_FAILED

:START_SUCCESS
popd
endlocal & exit /b 0

:MISSING_LAUNCHER
echo 找不到 launcher.pyw，请确认项目文件完整后重试。
pause
popd
endlocal & exit /b 2

:LAUNCH_FAILED
echo 无法启动本地桌面启动器，请检查 Python 环境后重试。
pause
popd
endlocal & exit /b 3
