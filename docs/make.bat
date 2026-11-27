@echo off
setlocal
pushd "%~dp0" || exit /b 1

rem Build the documentation from the source directory.
if not defined SPHINXBUILD set "SPHINXBUILD=sphinx-build"
set "SOURCEDIR=source"
set "BUILDDIR=build"
set "TARGET=%~1"
if not defined TARGET set "TARGET=help"

"%SPHINXBUILD%" -M %TARGET% "%SOURCEDIR%" "%BUILDDIR%" %SPHINXOPTS% %O%
set "BUILD_STATUS=%ERRORLEVEL%"
if "%BUILD_STATUS%"=="9009" echo python -m pip install -e ".[docs]"

popd
exit /b %BUILD_STATUS%
