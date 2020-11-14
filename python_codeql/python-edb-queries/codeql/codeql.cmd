@echo off
rem Wrapper provided for users who explicitly configured VS Code to point to codeql.cmd
"%~dp0\codeql.exe" %*
exit /b %errorlevel%
