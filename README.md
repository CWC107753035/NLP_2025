# Mental Health Sentiment Analysis from Reddit Posts

## Project Goal

This project aims to perform sentiment analysis on text data sourced from Reddit discussions related to mental health. The goal is to classify user posts or comments into predefined sentiment categories relevant to mental well-being, outputting soft labels (probabilities) for each category.

## Dataset

The primary dataset used for this project consists of text posts scraped from various mental health-related subreddits. The data includes the post text and associated labels indicating mental health status or sentiment.

* **Source:** CSV file (e.g., `Combined Data.csv`) containing Reddit posts.
* **Preprocessing:** Includes cleaning text, handling missing values, and encoding categorical labels into numerical format using `sklearn.preprocessing.LabelEncoder`.
* **Splitting:** The data is shuffled and split into training, validation, and test sets to ensure proper model evaluation.

*(Note: The raw data used for training is stored in the `/data/` directory, which is ignored by Git as specified in `.gitignore`)*

## Methodology

* **Model:** A Transformer-based model (e.g., DistilBERT, BERT, or a domain-specific variant like MentalBERT) pre-trained on a large text corpus is fine-tuned for the sequence classification task.
* **Framework:** TensorFlow (version 2.19+) with the Keras API.
* **Libraries:** Hugging Face `transformers` library for model loading and tokenization, `pandas` for data manipulation, `scikit-learn` for label encoding and evaluation metrics.
* **Output:** The model predicts probabilities (soft labels) for each of the defined sentiment classes.

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```
2.  **Create Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # Linux/macOS
    # venv\Scripts\activate # Windows
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    # Or install manually:
    # pip install tensorflow pandas transformers scikit-learn matplotlib seaborn
    ```
    *(Ensure your Python version and package versions are compatible. You might need specific versions of `numpy` based on your TensorFlow installation.)*

4.  **GPU Setup (if applicable):**
    * Ensure you have compatible NVIDIA drivers installed.
    * Install the correct versions of CUDA Toolkit and cuDNN library compatible with your TensorFlow version (refer to TensorFlow documentation).
    * Verify environment variables (like `LD_LIBRARY_PATH` on Linux) are set correctly or that necessary libraries are discoverable by TensorFlow.

## Usage

1.  **Data Preparation:**
    * Place your raw data CSV in the appropriate location (e.g., `/data/`).
    * Run the data loading and preprocessing script (e.g., `src/load_data.py`) to shuffle, split, tokenize, and create TensorFlow datasets. This script should also handle label encoding and save the fitted `LabelEncoder`.
2.  **Training / Fine-tuning:**
    * Run the training script (e.g., `src/train_model.py`). This script will:
        * Load the pre-trained Transformer model.
        * Load the prepared training and validation datasets.
        * Compile the model (optimizer, loss function).
        * Fine-tune the model using `model.fit()`.
        * (Optionally) Evaluate on the test set immediately after training.
        * Save the fine-tuned model and tokenizer to the `/model/` directory.
3.  **Evaluation:**
    * Run the evaluation script (e.g., `src/evaluate_model.py`). This script will:
        * Load the fine-tuned model and tokenizer from `/model/`.
        * Load the prepared test dataset.
        * Load the saved `LabelEncoder` to get class names.
        * Generate predictions on the test set.
        * Calculate and display metrics (Precision, Recall, F1-score, Confusion Matrix).

*(Note: The saved model files are stored in the `/model/` directory, which is ignored by Git as specified in `.gitignore`)*

## Directory Structure (Example)

.├── .gitignore├── README.md├── requirements.txt├── data/                # Contains raw/processed data (ignored by git)│   └── Combined Data.csv├── model/               # Contains saved model files (ignored by git)│   └── my_finetuned_model_full/│       └── ...└── src/                 # Source code├── load_data.py     # Script for data loading, preprocessing, splitting├── train_model.py   # Script for model training/fine-tuning└── evaluate_model.py # Script for model evaluation
## Future Work (Optional)

* Experiment with different pre-trained models (e.g., RoBERTa, MentalBERT).
* Hyperparameter tuning (learning rate, batch size, epochs).
* Explore more advanced text preprocessing techniques.
* Implement methods to handle class imbalance if present.
* Deploy the model as an API.
