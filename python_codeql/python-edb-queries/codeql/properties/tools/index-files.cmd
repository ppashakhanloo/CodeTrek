@echo off

type NUL && "%CODEQL_JAVA_HOME%\bin\java" ^
    -cp "%CODEQL_DIST%\tools\codeql.jar" ^
    com.semmle.cli.PropertiesExtractor ^
        --fileList="%1" ^
        --sourceArchiveDir="%CODEQL_EXTRACTOR_PROPERTIES_SOURCE_ARCHIVE_DIR%" ^
        --outputDir="%CODEQL_EXTRACTOR_PROPERTIES_TRAP_DIR%" ^
        --extensions=properties
exit /b %ERRORLEVEL%
