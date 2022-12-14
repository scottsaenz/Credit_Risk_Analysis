# Credit Risk Analysis
#### Author: Scott Saenz

## **Overview**
Credit risk analysis is used to determine which loans are at high-risk of defaulting. This is important for financial institutions because they make money off of the interest for loans, but lose money when a loan defaults. Identifying loans that are high risk of defaulting enables a bank to reach out to those customers and see if there are steps that can be taken to keep the account current. This analysis can also be used to determine if a line of credit should be approved or not, mitigating the risk that a financial institution takes on.
## **Results**
Two main types of analysis was conducted to determine if a loan is high risk or not, Ensemble Learners, which aggregate many small models to create a more significant model, and Resampling, which resamples the data so that the training set has about an equal chance of the different outcomes. This is used when there is a large difference between the two outcomes (high-risk and low-risk in this case).
<p>
Each model will have metrics that go along with it, accuracy, a confusion matrix, and classification report. Accuracy is the basic measurement of how many predictions were correct. With an imbalanced data set such, a dumb model could predict that every loan is low-risk. This would be a mistake as a true high-risk loan would result in greater financial ramifications than incorrectly labeling a low-risk loan. With that being said, models that have a low accuracy score are problematic because the end result would be incorrectly predicting both high-risk and low-risk loans.
</p>
<p>
A confusion matrix will quickly show the number of true positive, false positive, true negative, and false negative results. Visualizing this a quick way for different models to stand out and was the cause for eliminating all of the random sampling models from consideration.
</P>
<p>
A classification report is also included for each model. This provides more statistics that will help in evaluating models. This report includes precision, recall, specifity, f1, geometric mean, indexed balance accuracy, and support.

### **Ensemble Learners**
#### **Balanced Random Forest Classifier**
Accuracy of 72.5%

Confusion Matrix
A confusion matrix can be visuallized using the following code
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred, labels = model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=model.classes_)
disp.plot()
```


![Random Forest Confusion Matrix](RandomForestCM.png)

Classification Report Imbalanced
| | pre |      rec   |    spe  |      f1   |    geo  |     iba  |     sup |
|---|---|------------|---------|-----------|---------|----------|---------|
|  high_risk |      0.03 |     0.58 |     0.87 |     0.05 |     0.71 |     0.49 |      101 |
| low_risk |      1.00 |     0.87 |     0.58 |     0.93 |     0.71 |     0.52 |    17104|
| | | | | | | |
| avg / total |       0.99  |    0.86 |     0.59 |     0.92 |     0.71 |     0.52 |    17205 |

Most Signicant Features
| Significance | Feature |
| --- | --- | 
|0.076 | total_rec_prncp | 
|0.062 | last_pymnt_amnt | 
|0.060 | total_pymnt_inv |
|0.056 | total_rec_int |
|0.053 | total_pymnt |


#### **Easy Ensemble AdaBoost Classifier**
Accuracy of 93.2%

Confusion Matrix
![Easy Ensemble Confusion Matrix](EasyEnsembleCM.png)

Classification Report Imbalanced
| | pre |      rec |      spe |       f1  |     geo | iba |      sup |
|---|---|----------|----------|-----------|---------|-----|----------|
|  high_risk |      0.09  |    0.92 |     0.94 |     0.16 |     0.93  |    0.87 |      101 |
|   low_risk |      1.00 |     0.94 |     0.92 |     0.97 |     0.93  |    0.87 |     17104 |
|          | | | | | | | |
| avg / total |       0.99 |      0.94 |     0.92 |     0.97 |     0.93  |    0.87  |   17205

### **Resampling**
#### **Naive Random Oversampling**
Accuracy 65.1%

![Naive Random](NaiveRandomCM.png)

Classification Report Imbalanced

| |pre |      rec |      spe |        f1 |      geo |      iba |      sup |
|---|---|---------|----------|-----------|----------|----------|----------|
|  high_risk |      0.01 |     0.70 |     0.60 |     0.02 |     0.65 |     0.43 |      101
|   low_risk |      1.00 |     0.60   |   0.70 |     0.75  |    0.65 |     0.42 |    17104
|          | | | | | | | |
|avg / total |      0.99 |     0.60  |    0.70 |     0.74 |     0.65 |     0.42 |    17205


#### **SMOTE Oversampling**
Accuracy 67.3%

Confusion Matrix
![SMOTE](SMOTE_CM.png)

Classification Report Imbalanced

| |pre |  rec | spe |   f1  |  geo | iba | sup |
|---|---|-----|-----|-------|------|-----|-----|
|high_risk |      0.01 |  0.71 | 0.63 | 0.02 | 0.67 | 0.45  | 101
|low_risk | 1.00 |  0.63 | 0.71 | 0.77  |    0.67 |     0.45 |    17104
|          | | | | | | | |
|avg / total |      0.99 |     0.63  |    0.71 |     0.77  |    0.67 |     0.45  |   17205


#### **Undersampling**
Accuracy 53.8%

Confusion Matrix
![Undersampling](UndersamplingCM.png)

Classification Report Imbalanced
| |pre|       rec  | spe | f1 |   geo | iba  | sup |
|---|---|----------|-----|----|-------|------|-----|
|  high_risk  |     0.01  | 0.66 |  0.41 |  0.01  | 0.52 | 0.28 | 101|
|low_risk| 1.00 |     0.41   |   0.66  |    0.58 |     0.52  |    0.27  |   17104 |
|          | | | | | | | |
|avg | total  |     0.99  |    0.41  |    0.66  |    0.58    |  0.52   |   0.27  |   17205


#### **Combination Over/Under Sampling**
Accuracy 64.8%

Confusion Matrix
![Combination Over/Under Sampling](Combination_CM.png)

| | pre |      rec |      spe  |      f1  |     geo |      iba  |     sup |
|---|---|---|---|---|---|---|---|
|high_risk |      0.01 |     0.71 |     0.58 |     0.02 |     0.64 |     0.42 |      101|
|   low_risk |      1.00 |     0.58   |   0.71 |     0.74 |     0.64 |     0.41   |  17104
|          | | | | | | | |
|avg / total |      0.99 |     0.58 |     0.71  |    0.73 |     0.64 |     0.41 |    17205 |


## **Summary**
All of the resampling methods performed poorly and would have marked too many loans as high risk. There is a balance that needs to be met in order for a financial institution to be viable, and these methods would result in a strategy that is too risk averse.
<p>
The recommendation is to utilize the Easy Ensemble AdaBoost Classifier model to identify high-risk loans. It had a high accuracy for identifying both the high-risk and low-risk loans.

---
[Environment Configuration Ensemble Notebook](environment.yml)<p>
[Environment Configuration Resampling Notebook](resampling_env.yml)
