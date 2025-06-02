# Plan for Integrating Custom TensorFlow Invoice Model as a LangChain Tool

This document outlines the strategy for integrating a custom-trained TensorFlow model for invoice processing as a new tool within the existing LangChain agent framework.

## 1. Objective

The primary goal is to make the custom TensorFlow invoice processing model accessible and usable by the LangChain agent. This will allow the agent to leverage the custom model's capabilities for extracting key information from invoices, especially those with diverse or non-standard layouts, through natural language commands.

## 2. TensorFlow Model Deployment/Access Strategy

To make the TensorFlow model callable by the Python functions that will form the LangChain tool, we need a deployment or access strategy for the trained SavedModel.

*   **Options:**
    1.  **Local Loading:**
        *   Load the TensorFlow SavedModel directly within the Python function using `tf.saved_model.load()`.
        *   Suitable for initial development, testing, and single-instance deployments.
        *   Simplest to set up but not scalable for multiple users or high throughput.
    2.  **TensorFlow Serving:**
        *   Deploy the SavedModel using TensorFlow Serving, which provides a gRPC or REST API endpoint for inference.
        *   The Python function would then make API calls to this TF Serving endpoint.
        *   More scalable and suitable for production environments. Requires setting up TF Serving.
    3.  **Google Cloud AI Platform (Vertex AI) Prediction:**
        *   Upload the SavedModel to Google Cloud Storage and create a Vertex AI Endpoint.
        *   Vertex AI Prediction provides a fully managed, scalable HTTP endpoint for the model.
        *   The Python function would use the `google-cloud-aiplatform` client library to make predictions against this endpoint.
        *   Recommended for production due to scalability, management, and integration with the GCP ecosystem.

*   **Recommendation:**
    *   For initial development and ease of testing, **Local Loading** can be used.
    *   For production or more scalable deployments, **Vertex AI Prediction** is the recommended approach, followed by TensorFlow Serving if more custom server-side logic is needed around the model.
*   **Design Principle:** The core Python function for the tool should be designed with the assumption that it might interact with an API endpoint (e.g., taking image bytes and returning structured data). This makes transitioning from local loading to an API-based approach smoother.

## 3. Tool's Core Python Function (Example: `process_custom_invoice_tf_model`)

This function will encapsulate the logic for using the custom TensorFlow model. It would likely reside in `gcp_ai_services.py` or a new dedicated `custom_tf_models.py`.

*   **Function Signature (Conceptual):**
    ```python
    from typing import Dict, Any, Optional, Union, List
    import tensorflow as tf # If loading locally
    # from google.cloud import aiplatform # If using Vertex AI Prediction

    # Placeholder for model - would be loaded globally or per call for local,
    # or client initialized for endpoint.
    # LOCAL_MODEL_PATH = "path/to/your/tf_saved_model"
    # TF_MODEL = None
    # VERTEX_AI_ENDPOINT = None # Example: "projects/.../endpoints/..."

    def _preprocess_invoice_image_for_tf(image_path_or_bytes: Union[str, bytes]) -> Any:
        # TODO: Implement preprocessing based on the TF model's requirements
        # e.g., read image, resize, normalize, convert to tensor
        # This needs to match the preprocessing used during model training.
        # For LayoutLM-style models, this might also involve OCR and tokenization
        # if the model doesn't handle raw pixels + OCR internally.
        # For object detection, it's typically image resizing and normalization.
        print(f"Preprocessing image: {type(image_path_or_bytes)}")
        # Example (very basic, actual preprocessing will be model-specific):
        # img = tf.io.read_file(image_path) if isinstance(image_path_or_bytes, str) else image_path_or_bytes
        # img = tf.image.decode_image(img, channels=3)
        # img = tf.image.resize(img, [TARGET_HEIGHT, TARGET_WIDTH])
        # img = img / 255.0 # Normalize
        # return tf.expand_dims(img, axis=0) # Add batch dimension
        return "dummy_preprocessed_tensor_or_input_dict"


    def _postprocess_tf_model_output(model_output: Any) -> Dict[str, Any]:
        # TODO: Implement postprocessing based on the TF model's output format.
        # For LayoutLM (token classification): Convert token labels to field strings.
        # For Object Detection: Get bounding boxes, class labels, and scores.
        #    Then, apply OCR to these bounding boxes.
        #    Structure into a dictionary: {"invoice_id": "123", "total_amount": "100.00", ...}
        #    Include confidence scores if available.
        print(f"Postprocessing model output: {model_output}")
        return {"extracted_field_1": "value1_from_tf", "extracted_field_2": "value2_from_tf", "model_type": "custom_tensorflow"}

    def process_custom_invoice_tf_model(
        image_path: str, # Local path to the invoice image
        # model_endpoint_url: Optional[str] = None, # If using TF Serving or Vertex AI
        # local_model_path: Optional[str] = LOCAL_MODEL_PATH # If loading locally
    ) -> Optional[Dict[str, Any]]:
        """
        Processes an invoice image using the custom-trained TensorFlow model.

        Args:
            image_path: Path to the local invoice image file.
            # model_endpoint_url: Optional URL for a deployed model endpoint.
            # local_model_path: Optional path to a local SavedModel.

        Returns:
            A dictionary containing the extracted fields, or None if an error occurs.
        """
        global TF_MODEL # Or manage model loading state appropriately

        print(f"Received request to process custom invoice: {image_path}")
        if not os.path.exists(image_path):
            print(f"Error: Invoice file not found at {image_path}")
            return {"error": f"Invoice file not found at {image_path}"}

        try:
            # --- Option 1: Local Model Loading (Example) ---
            # if TF_MODEL is None and local_model_path:
            #     print(f"Loading custom TF model from: {local_model_path}")
            #     TF_MODEL = tf.saved_model.load(local_model_path) # Load once
            # if not TF_MODEL:
            #     return {"error": "Custom TF Model not loaded."}
            #
            # preprocessed_input = _preprocess_invoice_image_for_tf(image_path)
            # # This call depends on the actual signature of your SavedModel's serving function
            # raw_predictions = TF_MODEL.signatures["serving_default"](preprocessed_input) # Or specific signature
            # extracted_data = _postprocess_tf_model_output(raw_predictions)


            # --- Option 2: Vertex AI Prediction (Conceptual) ---
            # if model_endpoint_url: # This would actually be an endpoint resource name
            #    client_options = {"api_endpoint": "YOUR_REGION-aiplatform.googleapis.com"}
            #    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
            #    # Prepare instances (depends on model's expected input format)
            #    # e.g. for image bytes:
            #    with open(image_path, "rb") as f:
            #        image_bytes = f.read()
            #    encoded_content = base64.b64encode(image_bytes).decode("utf-8")
            #    instances = [{"image_bytes": {"b64": encoded_content}}] # Example
            #    response = client.predict(endpoint=model_endpoint_url, instances=instances)
            #    extracted_data = _postprocess_tf_model_output(response.predictions)


            # --- Placeholder for actual model call ---
            print("Simulating custom TF model processing...")
            preprocessed_input = _preprocess_invoice_image_for_tf(image_path)
            # Simulate model output
            simulated_raw_predictions = {"output_1": tf.constant([[0.9, 0.1, ...]]), "output_2": tf.constant([[...]])}
            extracted_data = _postprocess_tf_model_output(simulated_raw_predictions)
            # --- End Placeholder ---

            if not extracted_data:
                return {"error": "Custom TF model processing returned no data."}
            return extracted_data

        except Exception as e:
            print(f"Error during custom TF model processing for {image_path}: {e}")
            # import traceback; traceback.print_exc()
            return {"error": f"Exception during custom TF model processing: {str(e)}"}
    ```

## 4. LangChain Tool Definition

This involves creating a Pydantic schema for the tool's input and using the `@tool` decorator. This definition would go into `langchain_agent.py` or a dedicated `custom_tools.py` file.

*   **Pydantic Input Schema:**
    ```python
    from langchain.pydantic_v1 import BaseModel, Field

    class CustomInvoiceToolInput(BaseModel):
        file_path: str = Field(description="The local path to the invoice image file (e.g., PNG, JPG, PDF if model handles PDF directly, otherwise convert PDF to image first).")
        # Add other parameters if your core function or model needs them,
        # e.g., confidence_threshold: Optional[float] = Field(0.5, description="Confidence threshold for extractions.")
    ```

*   **Tool Definition using `@tool`:**
    ```python
    from langchain_core.tools import tool
    # Assuming process_custom_invoice_tf_model is imported from gcp_ai_services or custom_tf_models

    @tool(args_schema=CustomInvoiceToolInput)
    def custom_tensorflow_invoice_extractor(file_path: str) -> Dict[str, Any]:
        """
        Processes an invoice using a custom-trained TensorFlow model to extract key information.
        This tool is specialized for invoices that may not conform to standard templates
        and where a fine-tuned model is expected to perform better than generic OCR or form parsing.
        It typically extracts fields like invoice ID, vendor name, customer name, invoice date,
        due date, total amount, sub_total, tax amount, and line items if the model supports them.
        The input should be a local file path to an image of the invoice.
        Returns a dictionary of extracted fields and their values.
        """
        # Here, you would call the core Python function defined in step 3.
        # For example:
        # return gcp_ai_services.process_custom_invoice_tf_model(image_path=file_path)

        # Placeholder until the core function is fully implemented and model available:
        print(f"[Tool Placeholder] custom_tensorflow_invoice_extractor called with file: {file_path}")
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}. Please provide a valid local file path."}

        # Simulate a successful extraction for now
        return {
            "tool_used": "custom_tensorflow_invoice_extractor (simulated)",
            "file_processed": file_path,
            "simulated_extracted_fields": {
                "invoice_id": "SIM-TF-12345",
                "vendor_name": "Simulated TF Vendor",
                "total_amount": "99.99",
                "invoice_date": "2024-03-15"
            },
            "message": "This is a simulated output. The actual TensorFlow model integration is pending."
        }
    ```

## 5. Adding Tool to Agent

Once the tool is defined, it needs to be added to the `tools` list that is passed to the `AgentExecutor` in `langchain_agent.py`.

```python
# In langchain_agent.py

# ... other tool imports and definitions ...
# from custom_tools import custom_tensorflow_invoice_extractor # If in a separate file

tools = [
    process_document_tool,
    get_embedding_tool,
    read_text_file_tool,
    read_pdf_file_tool,
    fetch_web_page_content_tool,
    custom_tensorflow_invoice_extractor # Add the new tool
]

# ... rest of the agent setup ...
```

## 6. Testing Strategy

*   **Unit Testing the Core Function:**
    *   Test `process_custom_invoice_tf_model` (or its equivalent) with sample invoice images (and a dummy model if the real one isn't ready).
    *   Verify that preprocessing and postprocessing steps work as expected.
    *   Mock model inference if testing logic around it.
*   **LangChain Agent Integration Testing:**
    *   Start the `langchain_agent.py`.
    *   Formulate natural language queries that should logically trigger the `custom_tensorflow_invoice_extractor` tool. Examples:
        *   "Extract details from this invoice using the custom TensorFlow model: /path/to/my_invoice.png"
        *   "Use the special invoice processor for the file /path/to/another_invoice.jpg"
    *   **Examine `verbose=True` output:**
        *   Confirm the agent correctly identifies and selects the `custom_tensorflow_invoice_extractor` tool.
        *   Check that the `file_path` argument is correctly parsed from the user input and passed to the tool.
        *   Verify the (simulated or actual) output from the tool is returned to the agent and then presented to the user.
*   **Debugging Tool Description and Agent Prompt:**
    *   If the agent fails to pick the tool or picks the wrong one, the tool's docstring (description) might need refinement to be more distinctive or clearer.
    *   The main agent prompt might also need adjustments if there's consistent misinterpretation.
*   **End-to-End Testing (with a trained model):**
    *   Once a version of the custom TensorFlow model is trained and deployed/loadable, perform end-to-end tests with real invoice images.
    *   Compare extracted results against ground truth annotations.
    *   Evaluate success based on the accuracy of extracted fields.

## 7. Future Considerations (Optional)

*   **Handling Multiple Invoices/Batch Processing:** The current tool design focuses on a single `file_path`. For processing multiple invoices, the agent could be instructed to loop, or the tool itself could be enhanced to accept a list of file paths (though this makes the LangChain tool interface more complex).
*   **PDF to Image Conversion:** If the TF model primarily expects images, and users provide PDFs, the tool or a preceding helper tool might need to handle PDF to image conversion (e.g., using `pdf2image` library). This could be a separate tool or an internal step.
*   **Confidence Scores:** Include confidence scores from the TF model in the structured output for downstream applications to gauge reliability.
*   **Asynchronous Execution:** For models that take longer to run, consider making the tool execution asynchronous to avoid blocking the agent for too long (more advanced).

This plan provides a roadmap for integrating the custom TensorFlow model into the LangChain agent, enabling more powerful and specialized invoice processing capabilities.
