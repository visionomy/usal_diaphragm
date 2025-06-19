@echo off

rmdir /s /q build dist test
pyinstaller --noconfirm ..\src\usal_diaphragm\app\launch.py --onefile
if "%ERRORLEVEL%" neq "0" exit /b %ERRORLEVEL%

set ZIP="7z.exe"
set FNAME=usal_diaphragm.zip

if exist %FNAME% del %FNAME%

copy .\bat\*.bat dist\
copy ..\README.md dist\

cd dist
if exist launch.exe move launch.exe diaphragm.exe
diaphragm.exe animate_point_cloud --help > arguments.txt
cd ..

%ZIP% a %FNAME% .\dist\*

mkdir test
cd test
%ZIP% x ..\%FNAME%

cd ..
