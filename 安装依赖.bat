@echo off
chcp 65001 >nul
setlocal
pushd "%~dp0"

set "PYTHON_EXE="

:TRY_PYTHON
set "CANDIDATE_PYTHON="
for /f "tokens=1,* delims=#" %%A in (`python -c "import sys; sys.exit(3) if sys.version_info < (3, 10) else None; sys.stdout.reconfigure(encoding='utf-8', errors='strict'); print('CODEX_PYTHON_OK#' + sys.executable)" 2^>nul`) do (
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
for /f "tokens=1,* delims=#" %%A in (`py -3 -c "import sys; sys.exit(3) if sys.version_info < (3, 10) else None; sys.stdout.reconfigure(encoding='utf-8', errors='strict'); print('CODEX_PYTHON_OK#' + sys.executable)" 2^>nul`) do (
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
for /f "tokens=1,* delims=#" %%A in (`python3 -c "import sys; sys.exit(3) if sys.version_info < (3, 10) else None; sys.stdout.reconfigure(encoding='utf-8', errors='strict'); print('CODEX_PYTHON_OK#' + sys.executable)" 2^>nul`) do (
    if "%%A"=="CODEX_PYTHON_OK" (
        set "CANDIDATE_PYTHON=%%B"
    )
)
if defined CANDIDATE_PYTHON if exist "%CANDIDATE_PYTHON%" (
    set "PYTHON_EXE=%CANDIDATE_PYTHON%"
    goto :PYTHON_FOUND
)

:NO_PYTHON
echo 未找到可用的 Python 3.10+。建议安装 Python 3.10 或 3.11 后重新运行此文件。
pause
popd
endlocal & exit /b 1

:PYTHON_FOUND
if not exist "requirements.txt" goto :MISSING_REQUIREMENTS

echo.
echo 此操作需要联网，并会从所选软件包索引安装 requirements.txt 中的运行依赖。
echo 请确认网络连接正常后选择安装来源。

:MENU
echo.
echo 1. 使用当前 pip 配置
echo 2. 官方 PyPI
echo 3. 清华镜像
echo 0. 取消
set "CHOICE="
set /p "CHOICE=请输入选项："
if "%CHOICE%"=="1" goto :INSTALL_CURRENT
if "%CHOICE%"=="2" goto :INSTALL_OFFICIAL
if "%CHOICE%"=="3" goto :INSTALL_MIRROR
if "%CHOICE%"=="0" goto :CANCELLED
echo 输入无效，请重新选择。
goto :MENU

:INSTALL_CURRENT
"%PYTHON_EXE%" -m pip install -r requirements.txt
set "PIP_EXIT_CODE=%ERRORLEVEL%"
goto :INSTALL_RESULT

:INSTALL_OFFICIAL
"%PYTHON_EXE%" -m pip install -r requirements.txt -i https://pypi.org/simple
set "PIP_EXIT_CODE=%ERRORLEVEL%"
goto :INSTALL_RESULT

:INSTALL_MIRROR
"%PYTHON_EXE%" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
set "PIP_EXIT_CODE=%ERRORLEVEL%"
goto :INSTALL_RESULT

:INSTALL_RESULT
if not "%PIP_EXIT_CODE%"=="0" goto :INSTALL_FAILED

:INSTALL_SUCCESS
echo 依赖安装完成。现在可以运行“启动工作台.bat”。
popd
endlocal & exit /b 0

:INSTALL_FAILED
echo 安装失败。请检查网络，或重新运行后尝试其他镜像。
pause
popd
endlocal & exit /b %PIP_EXIT_CODE%

:CANCELLED
echo 已取消，未执行安装。
pause
popd
endlocal & exit /b 2

:MISSING_REQUIREMENTS
echo 找不到 requirements.txt，请确认项目文件完整后重试。
pause
popd
endlocal & exit /b 3
