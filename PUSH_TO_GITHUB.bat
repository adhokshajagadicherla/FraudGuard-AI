@echo off
echo ============================================
echo   PUSH TO GITHUB - SRM Fraud Detection
echo ============================================
echo.

set /p GITHUB_USER=Enter your GitHub username: 
set /p GITHUB_TOKEN=Enter your GitHub PAT token: 
set /p REPO_NAME=Repo name (press Enter for: credit-card-fraud-detection): 

if "%REPO_NAME%"=="" set REPO_NAME=credit-card-fraud-detection

echo.
echo [1/3] Creating GitHub repository...
curl -s -X POST https://api.github.com/user/repos ^
  -H "Authorization: token %GITHUB_TOKEN%" ^
  -H "Content-Type: application/json" ^
  -d "{\"name\":\"%REPO_NAME%\",\"description\":\"Credit Card Fraud Detection ML - SRM Tiruchirappalli\",\"private\":false}" > nul

echo [2/3] Setting remote and renaming branch...
git remote remove origin 2>nul
git remote add origin https://%GITHUB_USER%:%GITHUB_TOKEN%@github.com/%GITHUB_USER%/%REPO_NAME%.git
git branch -M main

echo [3/3] Pushing to GitHub...
git push -u origin main

echo.
echo ============================================
echo   DONE! Visit your repo at:
echo   https://github.com/%GITHUB_USER%/%REPO_NAME%
echo.
echo   DEPLOY on Streamlit Cloud:
echo   1. Go to https://share.streamlit.io
echo   2. New app ^> Select your repo
echo   3. Branch: main  ^|  File: app.py
echo   4. Click Deploy!
echo ============================================
pause
