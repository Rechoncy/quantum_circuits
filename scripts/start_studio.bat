@echo off
setlocal
for %%I in ("%~dp0..") do set "ROOT=%%~fI\"
set "PY_ENV=%ROOT%python_env"
if exist "%PY_ENV%\python.exe" (
    set "PATH=%PY_ENV%;%PY_ENV%\Library\bin;%PY_ENV%\Scripts;%PATH%"
    set "PYTHONNOUSERSITE=1"
    set "MPMATH_NOGMPY=1"
    "%PY_ENV%\python.exe" "%ROOT%run_studio.py" %*
) else (
    python "%ROOT%run_studio.py" %*
)
