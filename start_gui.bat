@echo off
setlocal
set "ROOT=%~dp0"
set "PY_ENV=%ROOT%python_env"
if exist "%PY_ENV%\python.exe" (
    set "PATH=%PY_ENV%;%PY_ENV%\Library\bin;%PY_ENV%\Scripts;%PATH%"
    set "PYTHONNOUSERSITE=1"
    set "MPMATH_NOGMPY=1"
    "%PY_ENV%\python.exe" "%ROOT%run_gui.py" %*
) else (
    set "PATH=C:\Software\miniconda3\envs\qutip;C:\Software\miniconda3\envs\qutip\Library\bin;C:\Software\miniconda3\envs\qutip\Scripts;%PATH%"
    set "MPMATH_NOGMPY=1"
    C:\Software\miniconda3\envs\qutip\python.exe "%ROOT%run_gui.py" %*
)
