@echo off

echo Starting output_stats

if exist "%~dp0\..\src\usal_diaphragm\app\launch.py" (
	set LAUNCH_CMD="%~dp0\..\venv\Scripts\python" "%~dp0\..\src\usal_diaphragm\app\launch.py"
) else (
	set LAUNCH_CMD="%~dp0\diaphragm.exe"
)

call %LAUNCH_CMD% ^
--action output_csv ^
--ztrim 350 --ytrim 30 ^
--peaks grad2d --peaks_axis 0 --grad_threshold 12.0 ^
--surface plane --n_surfaces_test 100 --n_surfaces_keep 10 ^
--sphere --cartesian ^
%*

if "%ERRORLEVEL%" neq "0" (
  pause
) else (
  echo Finished output_stats
)
