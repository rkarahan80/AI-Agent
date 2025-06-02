@echo off
REM Script to run the AI Agent on Windows

SETLOCAL

REM Check for essential environment variables
SET missing_vars=0
IF "%GOOGLE_APPLICATION_CREDENTIALS%"=="" (
    echo Warning: Environment variable GOOGLE_APPLICATION_CREDENTIALS is not set.
    SET missing_vars=1
)
IF "%GOOGLE_CLOUD_PROJECT%"=="" (
    echo Warning: Environment variable GOOGLE_CLOUD_PROJECT is not set.
    SET missing_vars=1
)
IF "%GOOGLE_CLOUD_BUCKET%"=="" (
    echo Warning: Environment variable GOOGLE_CLOUD_BUCKET is not set.
    SET missing_vars=1
)
IF "%GOOGLE_CLOUD_LOCATION%"=="" (
    echo Warning: Environment variable GOOGLE_CLOUD_LOCATION is not set (used by Vertex AI tools).
    SET missing_vars=1
)

IF %missing_vars%==1 (
    echo Please set the missing environment variables before running the agent.
    echo Refer to SETUP_GUIDE.md or USER_GUIDE.md for more information.
    REM Optionally, exit here:
    REM EXIT /B 1
)

echo Starting AI Agent...
python langchain_agent.py %*

ENDLOCAL
