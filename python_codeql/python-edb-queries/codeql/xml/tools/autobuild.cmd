@echo off

type NUL && "%CODEQL_DIST%\codeql" database index-files ^
    --include-extension=.xml ^
    --size-limit=5m ^
    --language=xml ^
    "%CODEQL_EXTRACTOR_XML_WIP_DATABASE%"

exit /b %ERRORLEVEL%
