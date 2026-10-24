@echo off
rem Windows counterpart of run-jupyter.sh: JupyterLab from the environment init.cmd created.
cd /d "%~dp0"
call .venv-tn\Scripts\activate.bat || exit /b 1
jupyter lab %*
