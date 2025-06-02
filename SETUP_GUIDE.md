# Google Cloud Setup Guide

This guide provides step-by-step instructions to set up your Google Cloud environment for using Vertex AI, Document AI, and Cloud Storage.

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

You need to enable the following APIs for your project:

*   **Vertex AI API**:
    *   Go to the [Vertex AI API page in the Cloud Console](https://console.cloud.google.com/apis/library/aiplatform.googleapis.com).
    *   Ensure your project is selected in the top menu bar.
    *   Click "ENABLE".
*   **Document AI API**:
    *   Go to the [Document AI API page in the Cloud Console](https://console.cloud.google.com/apis/library/documentai.googleapis.com).
    *   Ensure your project is selected.
    *   Click "ENABLE".
*   **Cloud Storage API**:
    *   The Cloud Storage API (storage.googleapis.com) is typically enabled by default for new projects. You can verify its status here: [Cloud Storage API page](https://console.cloud.google.com/apis/library/storage.googleapis.com).
    *   If it's not enabled, click "ENABLE".

## 3. Create a Service Account

Service accounts are used to grant permissions to your application to access Google Cloud services.

1.  **Go to Service Accounts**:
    *   In the Google Cloud Console, navigate to "IAM & Admin" > "Service Accounts".
    *   [Direct link to Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts).
    *   Ensure your project is selected.
2.  **Create Service Account**:
    *   Click "+ CREATE SERVICE ACCOUNT".
    *   **Service account name**: Enter a descriptive name (e.g., `vertex-docai-agent`). The Service account ID will be generated automatically.
    *   **Description**: Add an optional description.
    *   Click "CREATE AND CONTINUE".
3.  **Grant Permissions (Roles)**:
    *   In the "Grant this service account access to project" section, click "+ ADD ROLE" for each of the following roles:
        *   **Vertex AI User**: Search for `Vertex AI User` and select it. This role provides permissions to use Vertex AI resources.
        *   **Document AI Editor**: Search for `Document AI Editor` and select it. This allows creating and managing Document AI processors. If your agent only needs to *use* existing processors, `Document AI Viewer` might be sufficient, but `Editor` provides more flexibility.
        *   **Storage Object Admin**: Search for `Storage Object Admin` and select it. This grants full control over objects in Cloud Storage buckets.
    *   Click "CONTINUE".
4.  **Grant users access to this service account (Optional)**:
    *   You can skip this step for now.
    *   Click "DONE".

For more details, see [Creating service accounts](https://cloud.google.com/iam/docs/creating-managing-service-accounts).

## 4. Download the Service Account JSON Key File

The JSON key file is a secure credential that your application will use to authenticate as the service account.

1.  **Find your Service Account**:
    *   In the "Service Accounts" list, find the service account you just created.
2.  **Create Key**:
    *   Click on the three dots (Actions) menu next to the service account.
    *   Select "Manage keys".
    *   Click "ADD KEY" and choose "Create new key".
    *   Select "JSON" as the key type.
    *   Click "CREATE".
    *   A JSON file will be downloaded to your computer. **Store this file securely**, as it provides access to your Google Cloud resources. Do not commit it to your version control system.

For more details, see [Creating service account keys](https://cloud.google.com/iam/docs/creating-managing-service-account-keys).

## 5. Set Environment Variable for Credentials

Your application needs to know where to find the service account key file. You do this by setting the `GOOGLE_APPLICATION_CREDENTIALS` environment variable.

*   **Linux/macOS**:
    Open your terminal and run the following command, replacing `[PATH_TO_YOUR_KEY_FILE]` with the actual path to the JSON key file you downloaded:
    ```bash
    export GOOGLE_APPLICATION_CREDENTIALS="[PATH_TO_YOUR_KEY_FILE].json"
    ```
    To make this setting permanent, add this line to your shell's configuration file (e.g., `~/.bashrc`, `~/.zshrc`).

*   **Windows (Command Prompt)**:
    ```cmd
    set GOOGLE_APPLICATION_CREDENTIALS="[PATH_TO_YOUR_KEY_FILE].json"
    ```
*   **Windows (PowerShell)**:
    ```powershell
    $env:GOOGLE_APPLICATION_CREDENTIALS="[PATH_TO_YOUR_KEY_FILE].json"
    ```
    To make this setting permanent on Windows, you can set it through the Environment Variables system settings.

**Important**: You need to set this environment variable in the terminal session or environment where you will run your Python application.

For more information, see [Setting the credentials environment variable](https://cloud.google.com/docs/authentication/provide-credentials-adc#setting-the-environment-variable).

## 6. Install Python Client Libraries

You'll need the following Python client libraries to interact with the Google Cloud services. Install them using pip:

```bash
pip install google-cloud-aiplatform google-cloud-documentai google-cloud-storage google-api-python-client
```

*   `google-cloud-aiplatform`: For Vertex AI functionalities.
*   `google-cloud-documentai`: For Document AI processing.
*   `google-cloud-storage`: For interacting with Google Cloud Storage.
*   `google-api-python-client`: A general Google APIs client library that might be useful for other services or more advanced use cases.

It's recommended to use a virtual environment to manage your Python project dependencies.

---

You are now set up to start developing applications using Vertex AI, Document AI, and Cloud Storage!
Remember to keep your service account key file secure.
