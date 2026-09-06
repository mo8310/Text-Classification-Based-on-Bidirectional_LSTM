# Text Emotion Classification

A deep learning NLP project that classifies short pieces of text into one of four emotions — **anger, fear, joy, and sadness** — using a Bidirectional LSTM neural network built with TensorFlow/Keras.

## Overview

Given a sentence (e.g. *"I am pleased to say it was a wonderful project"*), the model predicts the underlying emotion and outputs a probability score for each class. The project covers the full pipeline: data loading, cleaning and class balancing, text tokenization, model building and training, evaluation, and inference on new text.

## Tech Stack

| Category | Tools / Libraries |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow / Keras |
| Data Handling | Pandas, NumPy |
| Preprocessing | Keras `Tokenizer`, `pad_sequences`, NLTK |
| Machine Learning Utilities | scikit-learn (`LabelEncoder`, `classification_report`, `confusion_matrix`) |
| Visualization | Matplotlib, Seaborn |
| Model Persistence | Pickle (tokenizer), Keras `.h5` (model) |
| Environment | Jupyter Notebook / Google Colab |

## Dataset

The project uses a labeled emotion dataset split into three files:
- `train.txt`
- `val.txt`
- `test.txt`

Each file is a `;`-delimited text file with two columns: `sentence` and `label`. The original dataset contains six emotion labels; **love** and **surprise** were dropped to focus on four well-represented classes: `anger`, `fear`, `joy`, and `sadness`.

## Project Workflow

### 1. Import Libraries
Load the core data science and visualization stack: Pandas, NumPy, Matplotlib, and Seaborn.

### 2. Load Data
Read the training, validation, and test sets from their respective `.txt` files into Pandas DataFrames.

### 3. Handle Data Distribution
- Remove the `love` and `surprise` classes from all three splits.
- Balance the remaining four classes (`joy`, `sadness`, `fear`, `anger`) via random sampling so the model isn't biased toward the majority class.
- Visualize label distribution before and after balancing with pie charts.

### 4. Encoding
Convert the text emotion labels into numeric form using scikit-learn's `LabelEncoder`, fitted on the training set and applied consistently to validation and test sets.

### 5. Tokenizer Building
- Build a Keras `Tokenizer` (vocabulary capped at 10,000 words) fitted on the training sentences.
- Convert sentences to padded integer sequences with `pad_sequences`.
- One-hot encode the labels using `to_categorical`.

### 6. Embedding & Model Architecture
A sequential neural network built with the following layers:

```
Embedding(input_dim=vocab_size, output_dim=128, input_length=100)
Bidirectional(LSTM(64, return_sequences=True))
Dropout(0.5)
GlobalMaxPooling1D()
Dense(64, activation='relu')
Dropout(0.5)
Dense(num_classes, activation='softmax')
```

- **Vocabulary size:** ~10,628
- **Embedding dimension:** 128
- **Sequence length:** 100
- **Total parameters:** ~1.47M

Compiled with:
- Optimizer: `Adam` (learning rate = 0.001)
- Loss: `categorical_crossentropy`
- Metrics: `accuracy`, `precision`, `recall`

Training uses `EarlyStopping` (monitoring validation loss, patience = 3) to prevent overfitting, run for up to 20 epochs with a batch size of 32.

### 7. Evaluation
The trained model is evaluated on the held-out test set, achieving:

| Metric | Score |
|---|---|
| Accuracy | ~94.1% |
| Precision | ~95.1% |
| Recall | ~93.7% |

A confusion matrix (via scikit-learn + Seaborn heatmap) is used to visualize per-class performance across the four emotions.

### 8. Model Saving
- The trained model is saved as `nlp.h5`.
- The fitted tokenizer is saved as `tokenizer.pkl` using Pickle, so both can be reloaded later without retraining.

### 9. Inference / Testing
A reusable `predict()` function:
1. Loads the saved model and tokenizer.
2. Converts new input text into a padded sequence.
3. Runs it through the model to get class probabilities.
4. Displays the predicted probability for each emotion as a horizontal bar chart.

Example:
```python
predict("I am afraid from her because she told me leave the office asap", "nlp.h5", "tokenizer.pkl")
# → high probability for "fear"
```

## How to Run

1. Place `train.txt`, `val.txt`, and `test.txt` in the working directory (or update the paths).
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow nltk
   ```
3. Run the notebook cells in order — data loading → balancing → encoding → tokenization → model training → evaluation → inference.
4. Use the saved `nlp.h5` and `tokenizer.pkl` files to classify new text without retraining.
