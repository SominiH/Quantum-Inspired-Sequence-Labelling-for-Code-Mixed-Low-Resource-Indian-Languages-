@echo off
echo =======================================================
echo   Push Project to GitHub
echo =======================================================
echo.
set /p REPO_URL="Enter your GitHub Repository URL (e.g. https://github.com/SominiH/quantum-sequence-labelling.git): "
if "%REPO_URL%"=="" (
    echo Error: No repository URL provided.
    pause
    exit /b
)

echo.
echo Setting remote origin to: %REPO_URL%
git remote remove origin 2>nul
git remote add origin %REPO_URL%

echo.
echo Pushing branch 'main' to GitHub...
git push -u origin main

echo.
if %ERRORLEVEL% EQU 0 (
    echo =======================================================
    echo   SUCCESS: Project pushed to GitHub successfully!
    echo =======================================================
) else (
    echo =======================================================
    echo   Push failed. Please verify your repository URL
    echo   and ensure you are signed in to GitHub.
    echo =======================================================
)
echo.
pause
