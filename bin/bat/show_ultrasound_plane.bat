@echo off

echo Starting make_us_movie

REM Add ffmpeg to the path so that it can be found by Python
REM SETLOCAL
REM set OLDPATH=%PATH%
REM setx PATH "%~dp0"\ffmpeg\bin;%PATH%

if exist "%~dp0\..\src\usal_diaphragm\app\launch.py" (
	set LAUNCH_CMD="%~dp0\..\venv\Scripts\python" "%~dp0\..\src\usal_diaphragm\app\launch.py"
) else (
	set LAUNCH_CMD="%~dp0\diaphragm.exe"
)

call %LAUNCH_CMD% ^
--action show_2d_movie ^
--ztrim 350 --ytrim 30 --show_mask ^
--show_peaks --peaks_axis 0 --peaks grad2d --grad_threshold 12.0 ^
--surface plane --n_surfaces_test 100 --n_surfaces_keep 10 ^
--n_slices 10 --rate 6 ^
%*

REM --ztrim 200 ^
REM --ytrim 50 ^
REM --surface plane ^
REM --n_surfaces_test 100 ^
REM --n_surfaces_keep 10 ^
REM --box ^
REM --cartesian ^
REM --filter ^

if "%ERRORLEVEL%" neq "0" (
  pause
) else (
    echo Finished show_2d_movie
)

REM setx PATH %OLDPATH%
REM set OLDPATH= 
REM ENDLOCAL
