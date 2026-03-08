@echo off
setlocal

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo Could not find vswhere.exe>&2
    exit /b 1
)

set "VSINSTALL="
for /f "usebackq tokens=*" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    set "VSINSTALL=%%I"
)

if not defined VSINSTALL (
    echo Could not locate a Visual Studio installation with C++ tools.>&2
    exit /b 1
)

pushd "%~dp0"
call "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 goto :build_failed

cmake -S . -B build -A x64
if errorlevel 1 goto :build_failed

cmake --build build --config Release
if errorlevel 1 goto :build_failed

popd
endlocal
exit /b 0

:build_failed
set "status=%ERRORLEVEL%"
popd
endlocal & exit /b %status%
