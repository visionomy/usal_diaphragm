@echo off

echo Starting show_surfaces

if exist "%~dp0\..\src\usal_diaphragm\app\launch.py" (
	set LAUNCH_CMD="%~dp0\..\venv\Scripts\python" "%~dp0\..\src\usal_diaphragm\app\launch.py"
) else (
	set LAUNCH_CMD="%~dp0\diaphragm.exe"
)

call %LAUNCH_CMD% ^
--action show_3d_points ^
--ztrim 350 --ytrim 30 --show_mask ^
--show_peaks --peaks_axis 0 --peaks grad2d --grad_threshold 12.0 ^
--surface plane --n_surfaces_test 100 --n_surfaces_keep 10 ^
--cartesian --box --rate 6 ^
%*

if "%ERRORLEVEL%" neq "0" (
  pause
) else (
  echo Finished show_surfaces
)
