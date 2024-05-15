set /p input=Do u remember to run clash?
set http_proxy=127.0.0.1:7890
set https_proxy=127.0.0.1:7890
git branch
git add .
git commit -m "push by bat %date%"
git push origin newBranch
set /p input=press any key to continue