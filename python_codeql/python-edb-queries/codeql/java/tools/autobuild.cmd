@echo off

type NUL && "%CODEQL_JAVA_HOME%\bin\java.exe" ^
    -jar "%CODEQL_EXTRACTOR_JAVA_ROOT%\tools\autobuild-fat.jar" ^
    autoBuild --no-indexing

exit /b %ERRORLEVEL%
