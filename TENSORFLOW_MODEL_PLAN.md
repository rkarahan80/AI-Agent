# TensorFlow Model Design and Development Plan for Custom Invoice Processing

This document outlines a model design and development plan using TensorFlow for building a custom solution to extract key information fields from diverse, non-standard invoices.

## 1. Objective

The primary objective is to develop a robust machine learning model capable of accurately identifying and extracting predefined key fields (e.g., invoice ID, total amount, vendor name) from a variety of invoice documents, including those with non-standard layouts and formats.

## 2. Proposed Model Architecture(s)

Given the nature of invoices (documents where layout and visual cues are as important as textual content), we propose the following architectures:

### 2.1. Primary Recommendation: Layout-Aware Transformer Models

*   **Examples:** LayoutLM family (LayoutLM, LayoutLMv2, LayoutLMv3), DiT (Document Image Transformer), Donut (Document Understanding Transformer), or similar models available via Hugging Face Transformers.
*   **Rationale:** These models are specifically designed for document understanding tasks. They integrate textual information (from OCR), visual information (from the document image), and layout/positional information (bounding box coordinates of words/tokens). This holistic approach is highly effective for invoices where the position of a field is a strong indicator of its identity.
*   **How it works:**
    *   They typically process document images and their corresponding OCRed text.
    *   Text embeddings, image embeddings (from a CNN backbone or ViT), and layout embeddings are combined.
    *   These combined representations are fed into a Transformer encoder.
    *   Task-specific heads are added on top for tasks like token classification (for field extraction, similar to Named Entity Recognition) or question answering.
*   **Advantages:** State-of-the-art performance on document AI tasks, can handle diverse layouts well, end-to-end training possible for some variants.
*   **Considerations:** Can be computationally intensive to train, requires careful data preparation to align text, image, and layout data.

### 2.2. Alternative/Comparison: Object Detection followed by OCR/NLP

*   **Examples:**
    *   **Object Detection Models:** EfficientDet, YOLO (via TensorFlow Hub or TensorFlow Object Detection API), Faster R-CNN.
    *   **OCR:** Google Cloud Vision API, Tesseract, or a custom OCR model.
    *   **NLP (Optional):** A simple rule-based parser or a small NLP model for refining extracted text.
*   **Rationale:** This is a more modular, two-stage approach. First, an object detection model identifies regions of interest (i.e., the bounding boxes of the target fields). Then, OCR is applied to these specific regions to extract the text.
*   **How it works:**
    1.  Train an object detection model to draw bounding boxes around each target field on the invoice image (e.g., a box for "invoice_id", another for "total_amount").
    2.  For each detected box, crop the image region.
    3.  Apply OCR to the cropped region to get the text.
    4.  (Optional) Post-process OCR results (e.g., data type conversion, normalization).
*   **Advantages:** Can leverage powerful pre-trained object detectors, might be simpler to set up initially if OCR is already robust, allows separate optimization of detection and OCR.
*   **Considerations:**
    *   **Error Propagation:** Errors in object detection (missed fields, incorrect boxes) will directly impact OCR and final extraction.
    *   **OCR Dependency:** Relies heavily on accurate OCR for the detected regions.
    *   **Context:** Less capable of using global document context compared to layout-aware transformers. For example, distinguishing between multiple dates might be harder.

**Decision Point:** While the Object Detection approach is viable, **Layout-Aware Transformers are generally recommended for achieving higher accuracy on complex and diverse invoices** due to their ability to jointly learn from text, image, and layout. This plan will primarily focus on the Layout-Aware Transformer approach, with notes on the Object Detection alternative.

## 3. Development Plan

### 3.1. Environment Setup
*   **Programming Language:** Python 3.x
*   **Core Libraries:**
    *   `tensorflow` (latest stable version)
    *   `transformers` (Hugging Face, for LayoutLM-family models)
    *   `tensorflow-datasets` (optional, for data loading/management)
    *   `tensorflow-text` (for text preprocessing if needed)
    *   `Pillow` or `OpenCV-Python` (for image manipulation)
    *   `pandas` (for handling annotation data)
    *   `scikit-learn` (for metrics and data splitting)
    *   (If using TFOD API for object detection): `tf-models-official`
*   **Hardware:** GPU (NVIDIA Tesla T4, V100, or A100) is highly recommended for training, especially for transformer models.
*   **Training Platforms (Considerations):**
    *   **Google Cloud AI Platform (Vertex AI Training):** Scalable, managed training service. Good for large datasets and distributed training.
    *   **Google Colaboratory (Colab Pro/Pro+):** Suitable for experimentation and smaller datasets with free/paid GPU access.
    *   **Local Machine:** Possible with a powerful GPU.

### 3.2. Data Preprocessing (Detailed)

This is a critical step and depends heavily on the chosen model architecture.

**For Layout-Aware Transformers (e.g., LayoutLM):**
1.  **Input:** Invoice images and corresponding OCR results (words and their bounding boxes).
    *   If OCR is not pre-existing, integrate an OCR step (e.g., Google Cloud Vision API, Tesseract) to get words and their coordinates.
2.  **Annotation Conversion:**
    *   Annotations (target field labels and their bounding boxes on the image) need to be converted into a format suitable for token classification or sequence labeling.
    *   This often involves:
        *   Matching annotated field boxes with OCRed word boxes.
        *   Assigning BIO (Beginning, Inside, Outside) tags or similar labels to each OCRed word/token based on whether it falls within an annotated field. E.g., a word "123" inside an "invoice_id" field might get a `B-INVOICE_ID` tag.
        *   For some models, this might be structured like a SQuAD (Stanford Question Answering Dataset) format, where each field is a question and the answer is a span of text.
3.  **Tokenization:**
    *   Use the tokenizer corresponding to the pre-trained LayoutLM model (e.g., from Hugging Face Transformers).
    *   Ensure bounding boxes from OCR are correctly aligned with the tokenized input. Models often expect bounding boxes at the sub-word token level.
4.  **Input Formatting:**
    *   Create input tensors: `input_ids`, `attention_mask`, `token_type_ids`, `bbox` (normalized coordinates of tokens), and `labels`.
    *   Normalize bounding box coordinates (e.g., to a 0-1000 scale or 0-1 relative to image size).
5.  **`tf.data` Pipeline:**
    *   Use `tf.data.Dataset` for efficient loading, preprocessing, shuffling, and batching of data.

**For Object Detection Models (e.g., using TFOD API):**
1.  **Input:** Invoice images.
2.  **Annotation Conversion:**
    *   Convert annotations (field labels and their bounding boxes) to TFRecord format, which is the standard input for the TensorFlow Object Detection API.
    *   Each record will contain the image and a list of bounding boxes with their corresponding class IDs (one class ID per target field type).
3.  **Image Preprocessing:** Resizing to a fixed input size, normalization (as defined by the chosen object detection model).
4.  **TFRecord Creation:** Generate `.tfrecord` files for training and validation splits.

### 3.3. Model Selection and Fine-tuning

**Layout-Aware Transformer (Primary Focus):**
1.  **Pre-trained Model Selection:**
    *   Choose a suitable pre-trained model from Hugging Face Transformers (e.g., `microsoft/layoutlmv3-base`, `naver-clova-ix/donut-base`). Consider models pre-trained on document-like data.
2.  **Task-Specific Head:**
    *   For field extraction, a token classification head (`TFBertForTokenClassification` or similar, adapted for layout models) is typically added on top of the base transformer.
    *   The number of output labels for this head will be `(num_target_fields * 2) + 1` (for B-FIELD, I-FIELD tags, plus O for Outside).
3.  **Optimizer:** AdamW (`tf.keras.optimizers.AdamW`) is a common choice for transformers.
4.  **Loss Function:** `tf.keras.losses.SparseCategoricalCrossentropy` (if labels are integers) or `CategoricalCrossentropy` (if labels are one-hot) for the token classification head.
5.  **Metrics:**
    *   Token-level accuracy, precision, recall, F1-score for each label.
    *   Entity-level (field-level) F1-score, precision, recall (more meaningful for the actual task). This requires post-processing to group tokens back into fields.
6.  **Training Loop:**
    *   Use Keras `model.fit()` or a custom training loop for more control.
    *   Implement learning rate scheduling (e.g., linear warmup and decay).
    *   Regularly save model checkpoints.
    *   Use callbacks for early stopping and TensorBoard logging.

**Object Detection Model (Alternative):**
1.  **Architecture Choice:** Select a model from the TensorFlow Object Detection API's model zoo (e.g., EfficientDet-D0, SSD ResNet50) or TensorFlow Hub.
2.  **Configuration (`pipeline.config`):**
    *   Modify the `pipeline.config` file for the chosen model.
    *   Set paths to data (TFRecords), label map (`label_map.pbtxt` defining field names and their IDs).
    *   Configure batch size, learning rate, number of training steps, image resizing parameters, and augmentation options.
3.  **Training:** Use the `model_main_tf2.py` script (for TFOD API) or custom training script to train the model.
4.  **Post-OCR and Text Processing:** After getting bounding boxes from the object detector, apply OCR to those regions and implement logic to clean/normalize the extracted text.

### 3.4. Experimentation and Evaluation
*   **Hyperparameter Tuning:** Experiment with learning rates, batch sizes, number of epochs, optimizer parameters, and potentially different pre-trained model backbones.
*   **Validation Set Evaluation:** Continuously monitor performance on the validation set during training to prevent overfitting and guide hyperparameter choices.
*   **Test Set Evaluation:** Perform a final, unbiased evaluation on the held-out test set.
*   **Field-Specific Metrics:** Calculate precision, recall, and F1-score for each individual field (e.g., F1 for `invoice_id`, F1 for `total_amount`). This helps identify which fields are harder for the model.
*   **Error Analysis:**
    *   Manually review predictions on the validation/test set where the model fails.
    *   Categorize errors (e.g., missed field, incorrect boundary, misclassified field, OCR error).
    *   Use insights from error analysis to guide further data collection (e.g., more examples of a difficult layout), annotation refinement, or model adjustments.

### 3.5. Model Saving and Export
*   **Format:** Save the trained model in TensorFlow's SavedModel format. This format is suitable for deployment using TensorFlow Serving, AI Platform Prediction, or conversion to TensorFlow Lite.
*   **Inference Graph:** Ensure the SavedModel includes a serving signature for easy inference.
*   **For Object Detection (TFOD API):** Export the trained model using the `exporter_main_v2.py` script.

## 4. Iterative Approach

Model development is rarely a linear process. Expect to iterate:
1.  **Baseline Model:** Start with a reasonable pre-trained model and a subset of your data to establish a baseline performance.
2.  **Analyze and Refine:** Based on the baseline's performance and error analysis:
    *   Collect more diverse or targeted data if needed.
    *   Refine annotation guidelines and quality.
    *   Experiment with different model architectures or hyperparameters.
    *   Improve preprocessing or post-processing logic.
3.  **Scale Up:** Gradually increase data volume and model complexity as you see improvements.

This iterative approach ensures efficient use of resources and helps in building a progressively better model.
