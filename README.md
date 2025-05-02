# Mental Health Sentiment Analysis from Reddit Posts

## Project Goal

This project aims to perform sentiment analysis on text data sourced from Reddit discussions related to mental health. The goal is to classify user posts or comments into predefined sentiment categories relevant to mental well-being, outputting soft labels (probabilities) for each category.

## Dataset

The primary dataset used for this project consists of text posts scraped from various mental health-related subreddits. The data includes the post text and associated labels indicating mental health status or sentiment.

* **Source:** https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health
* **Preprocessing:** Includes cleaning text, handling missing values, and encoding categorical labels into numerical format using `sklearn.preprocessing.LabelEncoder`.
* **Splitting:** The data is shuffled and split into training, validation, and test sets to ensure proper model evaluation.

## Methodology

* **Model:** Use DistilBERT model, pre-trained on a large text corpus is fine-tuned for the sequence classification task.
* **Framework:** TensorFlow (version 2.19+) with the Keras API.
* **Libraries:** Hugging Face `transformers` library for model loading and tokenization, `pandas` for data manipulation, `scikit-learn` and `matplotlib` for label encoding and evaluation metrics.
* **Output:** The model predicts probabilities (soft labels) for each of the defined sentiment classes.

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/CWC107753035/NLP_2025.git
    cd <your-repo-directory>
    ```
2.  **Create Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # Linux/macOS
    # venv\Scripts\activate # Windows
    ```
3.  **Versions**
    
    TensorFlow Version: TensorFlow 2.19.0.

    CUDA Toolkit Version: CUDA 12.3

    cuDNN Version: cuDNN 8.9.7 
    
5.  **GPU Setup (if applicable):**
    * Ensure you have compatible NVIDIA drivers installed.
    * Install the correct versions of CUDA Toolkit and cuDNN library compatible with your TensorFlow version (refer to TensorFlow documentation).
    * Verify environment variables (like `LD_LIBRARY_PATH` on Linux) are set correctly or that necessary libraries are discoverable by TensorFlow.

## Usage

1.  **Data Preparation:**
    * Place your raw data CSV in the appropriate location and modify the notebook regarding path.
    * The data loading and preprocessing script will shuffle, split, tokenize, and create TensorFlow datasets. This script should also handle label encoding and save the fitted `LabelEncoder`.
2.  **Training / Fine-tuning:**
    The training part will:
        * Load the pre-trained Transformer model.
        * Load the prepared training and validation datasets.
        * Compile the model (optimizer, loss function).
        * Fine-tune the model using `model.fit()`.
3.  **Evaluation:**
    * The evaluation part will:
        * Load the fine-tuned model and tokenizer from `/model/`.
        * Load the prepared test dataset.
        * Load the saved `LabelEncoder` to get class names.
        * Generate predictions on the test set.
        * Calculate and display metrics (Precision, Recall, F1-score, Confusion Matrix).

## Future Work

* Experiment with different pre-trained models (e.g., RoBERTa, MentalBERT).
* Hyperparameter tuning (learning rate, batch size, epochs).
* Explore more advanced text preprocessing techniques.
* Implement methods to handle class imbalance if present.
