# Logistic Regression Baseline Evaluation

## Model Overview

For this project, I used Logistic Regression as the baseline model for detecting fraudulent credit card transactions. The model was trained using the processed training data and then evaluated using the test data. Since the dataset is highly imbalanced and less than 1% of the transactions are fraudulent, I focused on precision, recall, and F1 score in addition to accuracy.

## Evaluation Results

- Accuracy: 88.19%
- Precision: 2.38%
- Recall: 73.89%
- F1 Score: 4.61%

### Confusion Matrix

- True Negatives: 488,505
- False Positives: 65,069
- False Negatives: 560
- True Positives: 1,585

## Evaluation

The Logistic Regression model had an accuracy of 88.19%, but accuracy alone does not give the best picture of the model's performance because there are significantly more legitimate transactions than fraudulent transactions in the dataset. The model had a recall of 73.89%, meaning it was able to correctly identify about 74% of the actual fraudulent transactions. However, the precision was only 2.38%, meaning many legitimate transactions were also incorrectly identified as fraud. This is also reflected in the 65,069 false positives in the confusion matrix.

Overall, Logistic Regression provides a useful baseline for the project. The model was able to detect a large portion of the fraudulent transactions, but the high number of false positives shows that there is room for improvement. These results can be used as a benchmark when comparing Logistic Regression with the other machine learning models used in the project.

## Visualization

The Precision, Recall, and F1 scores are shown in:

`figures/logistic_regression_metrics.png`