@echo off

type NUL && "%CODEQL_JAVA_HOME%\bin\java" ^
    -jar "%CODEQL_EXTRACTOR_CSV_ROOT%\tools\csv-extractor.jar" ^
        --fileList="%1" ^
        --sourceArchiveDir="%CODEQL_EXTRACTOR_CSV_SOURCE_ARCHIVE_DIR%" ^
        --outputDir="%CODEQL_EXTRACTOR_CSV_TRAP_DIR%"

exit /b %ERRORLEVEL%
