# Sentiment Analysis with Logistic Regression (SAWLR)

## Overview

This project implements a sentiment analysis system using Logistic Regression with TF-IDF text vectorization. It classifies text reviews as positive (1) or negative (0) sentiment. The notebook demonstrates a complete workflow from data preparation to model evaluation.

## Features

- **Text Vectorization**: Uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert text to numerical features
- **Sentiment Classification**: Logistic Regression model for binary classification
- **Model Evaluation**: Provides precision, recall, and F1-score metrics
- **Example Prediction**: Includes a demonstration of predicting sentiment for new text

## Requirements

- Python 3.x
- pandas
- scikit-learn
- NumPy (automatically installed with scikit-learn)

## Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd sentiment-analysis
   ```

2. Install the required packages:
   ```bash
   pip install pandas scikit-learn
   ```

## Usage

1. Run the Jupyter notebook:
   ```bash
   jupyter notebook SAWLR.ipynb
   ```

2. The notebook will:
   - Create a sample dataset of product reviews
   - Convert text to TF-IDF features
   - Train a Logistic Regression model
   - Evaluate model performance
   - Demonstrate a sample prediction

## Sample Output

```
              precision    recall  f1-score   support

           0       0.00      0.00      0.00         1
           1       0.50      1.00      0.67         1

    accuracy                           0.50         2
   macro avg       0.25      0.50      0.33         2
weighted avg       0.25      0.50      0.33         2

Prediction for 'I like it': [1]
```

## Customization Options

- **Dataset**: Replace the sample data with your own labeled dataset
- **Vectorization**: Adjust TF-IDF parameters (`max_features`, `ngram_range`, etc.)
- **Model**: Try different classifiers (SVM, Random Forest, etc.)
- **Evaluation**: Add cross-validation or different metrics

## Important Notes

1. The warning messages indicate some metrics are undefined due to the very small sample size
2. For production use, you should:
   - Use a larger, more balanced dataset
   - Consider more sophisticated text preprocessing
   - Implement proper train-test-validation splits
   - Add hyperparameter tuning

## Future Improvements

- Add text preprocessing (stemming, lemmatization, stopword removal)
- Implement more advanced models (BERT, LSTMs)
- Create a pipeline for easy deployment
- Build a web interface for interactive predictions
- Add confidence scores to predictions

## License

This project is open-source and available under the MIT License.
