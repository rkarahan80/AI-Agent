# Google Cloud Setup Guide

This guide provides step-by-step instructions to set up your Google Cloud environment for using the AI Agent, which leverages Vertex AI, Document AI, and Cloud Storage.

## 1. Create or Select a Google Cloud Project

*   **New Project**: If you don't have an existing project, create a new one:
    *   Go to the [Google Cloud Console](https://console.cloud.google.com/).
    *   Click on the project selector dropdown in the top menu bar.
    *   Click on "NEW PROJECT".
    *   Enter a project name, select a billing account, and choose an organization or location if applicable.
    *   Click "CREATE".
*   **Existing Project**: If you have an existing project you'd like to use:
    *   Go to the [Google Cloud Console](https://console.cloud.google.com/).
    *   Select your project from the project selector dropdown in the top menu bar.

For more details, see [Creating and managing projects](https://cloud.google.com/resource-manager/docs/creating-managing-projects).

## 2. Enable Required APIs

You need to enable the following APIs for your project. Ensure your project is selected in the Cloud Console before enabling each API.

*   **Vertex AI API**:
    *   Go to the [Vertex AI API page in the Cloud Console](https://console.cloud.google.com/apis/library/aiplatform.googleapis.com).
    *   Click "ENABLE".
*   **Document AI API**:
    *   Go to the [Document AI API page in the Cloud Console](https://console.cloud.google.com/apis/library/documentai.googleapis.com).
    *   Click "ENABLE".
*   **Cloud Storage API**:
    *   The Cloud Storage API (storage.googleapis.com) is typically enabled by default for new projects. You can verify its status here: [Cloud Storage API page](https://console.cloud.google.com/apis/library/storage.googleapis.com).
    *   If it's not enabled, click "ENABLE".

It may take a few minutes for the APIs to be fully enabled.

## 3. Create a Service Account

Service accounts are used to grant permissions to your application to access Google Cloud services without using your personal user credentials.

1.  **Go to Service Accounts**:
    *   In the Google Cloud Console, navigate to "IAM & Admin" > "Service Accounts".
    *   [Direct link to Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts).
    *   Ensure your project is selected.
2.  **Create Service Account**:
    *   Click "+ CREATE SERVICE ACCOUNT".
    *   **Service account name**: Enter a descriptive name (e.g., `ai-agent-service-account`). The Service account ID will be generated automatically.
    *   **Description**: Add an optional description (e.g., "Service account for AI Document Processing Agent").
    *   Click "CREATE AND CONTINUE".
3.  **Grant Permissions (Roles)**:
    *   In the "Grant this service account access to project" section, click "+ ADD ROLE" for each of the following roles:
        *   **Vertex AI User**: Search for `Vertex AI User` and select it. This role provides permissions to use Vertex AI resources, including making predictions with models.
        *   **Document AI Editor**: Search for `Document AI Editor` and select it. The `Document AI Editor` role allows the agent to both manage processors (if needed in future extensions) and process documents with existing ones.
        *   **Storage Object Admin**: Search for `Storage Object Admin` and select it. This grants full control over objects in Cloud Storage buckets, which the agent uses for storing and managing files.
    *   Click "CONTINUE".
4.  **Grant users access to this service account (Optional)**:
    *   This step is generally not required for the agent's operation. You can skip it.
    *   Click "DONE".

For more details, see [Creating service accounts](https://cloud.google.com/iam/docs/creating-managing-service-accounts).

## 4. Download the Service Account JSON Key File

The JSON key file is a secure credential that your application will use to authenticate as the service account.

1.  **Find your Service Account**:
    *   In the "Service Accounts" list in the IAM & Admin section, find the service account you just created.
2.  **Create Key**:
    *   Click on the three dots (Actions) menu next to the service account.
    *   Select "Manage keys".
    *   Click "ADD KEY" and choose "Create new key".
    *   Select "JSON" as the key type.
    *   Click "CREATE".
    *   A JSON file will be downloaded to your computer. **Store this file securely**, as it provides access to your Google Cloud resources. Do not commit it to your version control system (e.g., Git). Consider adding its name to your project's `.gitignore` file.

For more details, see [Creating service account keys](https://cloud.google.com/iam/docs/creating-managing-service-account-keys).

## 5. Set Environment Variables for Configuration and Credentials

Your application needs to know where to find the service account key file and other important configuration details like your Project ID and default location for services. You do this by setting environment variables.

**Key Environment Variables:**

*   `GOOGLE_APPLICATION_CREDENTIALS`: Path to your service account JSON key file.
    *   **Linux/macOS (bash/zsh)**:
        ```bash
        export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/downloaded-key-file.json"
        ```
        To make this setting permanent, add this line to your shell's configuration file (e.g., `~/.bashrc`, `~/.zshrc`).
    *   **Windows (Command Prompt)**:
        ```cmd
        set GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\your\downloaded-key-file.json"
        ```
    *   **Windows (PowerShell)**:
        ```powershell
        $env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\your\downloaded-key-file.json"
        ```
        To make this setting permanent on Windows, you can set it through the Environment Variables system settings panel.

*   `GOOGLE_CLOUD_PROJECT`: Your Google Cloud Project ID. This is used by various Google Cloud client libraries and the agent to identify your project.
    *   **Linux/macOS (bash/zsh)**: `export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"`
    *   **Windows (Command Prompt)**: `set GOOGLE_CLOUD_PROJECT="your-gcp-project-id"`
    *   **Windows (PowerShell)**: `$env:GOOGLE_CLOUD_PROJECT="your-gcp-project-id"`

*   `GOOGLE_CLOUD_LOCATION`: The default Google Cloud region for services like Vertex AI (e.g., "us-central1").
    *   **Linux/macOS (bash/zsh)**: `export GOOGLE_CLOUD_LOCATION="us-central1"`
    *   **Windows (Command Prompt)**: `set GOOGLE_CLOUD_LOCATION="us-central1"`
    *   **Windows (PowerShell)**: `$env:GOOGLE_CLOUD_LOCATION="us-central1"`

The `USER_GUIDE.md` also lists `GOOGLE_CLOUD_BUCKET` which you'll need to set for the agent to know which Cloud Storage bucket to use.

**Important**: You need to set these environment variables in the terminal session or environment where you will run your Python application.

For more information, see [Setting the credentials environment variable](https://cloud.google.com/docs/authentication/provide-credentials-adc#setting-the-environment-variable) and related Google Cloud documentation on client library configuration.

## 6. Install Python Client Libraries

The project uses several Python client libraries to interact with Google Cloud services and for other functionalities. After cloning the project repository, navigate to its root directory and install all dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

This will install all necessary libraries, including:
*   `google-cloud-aiplatform`: For Vertex AI functionalities.
*   `google-cloud-documentai`: For Document AI processing.
*   `google-cloud-storage`: For interacting with Google Cloud Storage.
*   `langchain`, `langchain-google-vertexai`, and other LangChain packages: For the agent framework.
*   And other supporting libraries like `pdfplumber`, `requests`, `beautifulsoup4`.

It's highly recommended to use a Python virtual environment (e.g., `venv`) to manage your project dependencies and avoid conflicts with system-wide packages.
To create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
Then run `pip install -r requirements.txt` within the activated environment.

---

You are now set up to start developing with and using the AI Agent! Remember to keep your service account key file secure. Refer to `USER_GUIDE.md` for instructions on running and interacting with the agent.
