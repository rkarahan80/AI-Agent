#!/bin/bash

# Script to run the AI Agent

# Check for essential environment variables
missing_vars=0
if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "Warning: Environment variable GOOGLE_APPLICATION_CREDENTIALS is not set."
    missing_vars=1
fi
if [ -z "$GOOGLE_CLOUD_PROJECT" ]; then
    echo "Warning: Environment variable GOOGLE_CLOUD_PROJECT is not set."
    missing_vars=1
fi
if [ -z "$GOOGLE_CLOUD_BUCKET" ]; then
    echo "Warning: Environment variable GOOGLE_CLOUD_BUCKET is not set."
    missing_vars=1
fi
if [ -z "$GOOGLE_CLOUD_LOCATION" ]; then
    echo "Warning: Environment variable GOOGLE_CLOUD_LOCATION is not set (used by Vertex AI tools)."
    missing_vars=1
fi

if [ "$missing_vars" -eq 1 ]; then
    echo "Please set the missing environment variables before running the agent."
    echo "Refer to SETUP_GUIDE.md or USER_GUIDE.md for more information."
    # Optionally, exit here if you want to be strict:
    # exit 1
fi

echo "Starting AI Agent..."
# Ensure python3 is used if python might point to python2
# Use python3 explicitly if available, otherwise fall back to python
PYTHON_CMD="python"
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
fi

$PYTHON_CMD langchain_agent.py "$@"
