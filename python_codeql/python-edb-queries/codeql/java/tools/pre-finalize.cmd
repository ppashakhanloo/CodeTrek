@echo off

IF [%LGTM_INDEX_XML_MODE%]==[] SET LGTM_INDEX_XML_MODE=default

IF [%LGTM_INDEX_XML_MODE%]==[default] (
type NUL && "%CODEQL_DIST%\codeql" database index-files ^
    --include "**/AndroidManifest.xml" ^
    --include "**/pom.xml" ^
    --include "**/web.xml" ^
    --size-limit 10m ^
    --language xml ^
    -- ^
    "%CODEQL_EXTRACTOR_JAVA_WIP_DATABASE%"
) ELSE IF [%LGTM_INDEX_XML_MODE%]==[smart] (
SET "CODEQL_EXTRACTOR_XML_PRIMARY_TAGS=faceted-project project plugin idea-plugin beans struts web-app module ui:UiBinder persistence"
type NUL && "%CODEQL_DIST%\codeql" database index-files ^
    --include-extension=.xml ^
    --size-limit 10m ^
    --language xml ^
    -- ^
    "%CODEQL_EXTRACTOR_JAVA_WIP_DATABASE%"
) ELSE IF [%LGTM_INDEX_XML_MODE%]==[all] (
type NUL && "%CODEQL_DIST%\codeql" database index-files ^
    --include-extension=.xml ^
    --size-limit 10m ^
    --language xml ^
    -- ^
    "%CODEQL_EXTRACTOR_JAVA_WIP_DATABASE%"
)

IF %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

IF [%LGTM_INDEX_PROPERTIES_FILES%]==[true] ^
type NUL && "%CODEQL_DIST%\codeql" database index-files ^
    --include-extension=.properties ^
    --size-limit=5m ^
    --language properties ^
    -- ^
    "%CODEQL_EXTRACTOR_JAVA_WIP_DATABASE%"

exit /b %ERRORLEVEL%
