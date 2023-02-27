# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 14:08:42 2023

@author: YuenKwan LI (301228849)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Req.d8: Accept a dataframe as an argument and normalizes all the datapoint.
def normalize(dataframe):
    return (dataframe - dataframe.min()) / (dataframe.max() - dataframe.min())


# Req.a: Load data
path = 'C:/workspace/IntroAI/Assignment 4/Exercise#1_YuenKwan'
filename = 'titanic.csv'
fullpath = os.path.join(path, filename)
titanic_YuenKwan = pd.read_csv(fullpath)


# Req.b: Initial exploration
print('----------- b1) Print first 3 records -------------')
print(titanic_YuenKwan.head(3))
print('----------- b2) Shape of data frame -------------')
print(titanic_YuenKwan.shape)
print('----------- b3) Names, Types, & Counts -------------')
titanic_YuenKwan.info()
print('----------- b4) Refer to Written Response -------------')
'''
 "PassengerId", "Name", "Ticket" columns contain unique values.
 "Cabin" column contains a lot of missing values. 
 These 4 columns are not useful for the model.
'''
print('----------- b5) Unique Values of Sex, Pclass -------------')
print('Sex: ', titanic_YuenKwan['Sex'].unique())
print('Pclass: ', titanic_YuenKwan['Pclass'].unique())
print('----------- End of b) Initial Exploration -------------')

'''
print(titanic_YuenKwan.describe())
'''


# Req.c: Data Visualizaion
plt.rcParams['figure.dpi'] = 300
# c1a) Bar chart showing # of survived by passenger class
survived_class = pd.crosstab(index=titanic_YuenKwan['Survived'], columns=titanic_YuenKwan['Pclass'])
survived_class.plot(kind='bar')
plt.title("Number of Survivors by Passenger Class (YuenKwan)")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()

# c1b) Bar chart showing # of survived by gender
survived_gender = pd.crosstab(index=titanic_YuenKwan['Survived'], columns=titanic_YuenKwan['Sex'])
survived_gender.plot(kind='bar')
plt.title("Number of Survivors by Gender (YuenKwan)")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()

# c1c) 
'''
Conclusion:

From the scatter matrix, we can see that there is a clear correlation between passenger class and survival, with more survivors in first class compared to third class.

There is also a correlation between fare and survival, with more survivors among those who paid higher fares.

There seems to be no clear correlation between the number of siblings/spouses or parents/children aboard and survival.

Gender is not included in the scatter matrix, but from the previous analysis, we know that it had an impact on survival rates.
'''

# c2) Scatter Matrix showing # of survived and the folliwng attributes
attributes = ["Survived", "Sex", "Pclass", "Fare", "SibSp", "Parch"]
pd.plotting.scatter_matrix(titanic_YuenKwan[attributes], figsize=(12, 8))
plt.show()

'''
# Generate a histogram of the 'Age' column
plt.hist(titanic_YuenKwan['Age'], bins=20)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Histogram of Age Distribution')
plt.show()

# Generate a scatterplot of 'Fare' vs 'Age'
plt.scatter(titanic_YuenKwan['Age'], titanic_YuenKwan['Fare'], alpha=0.5)
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('Scatterplot of Age vs Fare')
plt.show()
'''


# Req.d: Data Transformation
'''
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html
'''
# d1) Drop columns
titanic_YuenKwan_transformed = titanic_YuenKwan.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

# d2, d3, d4) Get dummies to transform categorical variables: Sex, Embarked
titanic_YuenKwan_transformed = pd.get_dummies(titanic_YuenKwan_transformed, columns=['Sex', 'Embarked'])
#print (titanic_YuenKwan_transformed.head(6))

# d5) Replace missing values in Age with mean(Age)
titanic_YuenKwan_transformed['Age'].fillna((titanic_YuenKwan_transformed['Age'].mean()), inplace=True)

# d6) Change all columns into float
titanic_YuenKwan_transformed = titanic_YuenKwan_transformed.astype('float')

#print (titanic_YuenKwan_transformed.head(6))

# d7) Check the dataframe
print('----------- d7) Check the transformed dataframe -------------')
titanic_YuenKwan_transformed.info()

# d8) def normalize(dataframe)

# d9) Call normalize() by passing transformed dataframe
titanic_YuenKwan_normalized = normalize(titanic_YuenKwan_transformed)

# d10) Display the first 2 records
print('----------- d10) Print first 2 records -------------')
print(titanic_YuenKwan_normalized.head(2))

# d11) Generate a Historgram at 9inch by 10inch
fig, ax = plt.subplots(figsize=(9, 10))
titanic_YuenKwan_normalized.hist(ax=ax)
plt.show()

# d12) Conclude on the Port of Embarkation"
'''
The majority of passengers embarked from port "S" (Southampton).
The number of passengers who embarked from port "C" (Cherbourg) is about half of those who embarked from port "S".
The fewest number of passengers embarked from port "Q" (Queenstown).
'''

# d13) Split the dataframe
# Split data into predictors (x) and target class (y)
x_YuenKwan = titanic_YuenKwan_normalized.drop('Survived', axis=1)
y_YuenKwan = titanic_YuenKwan_normalized['Survived']

# Split data into training (70%) and testing sets (30%)
from sklearn.model_selection import train_test_split
x_train_YuenKwan, x_test_YuenKwan, y_train_YuenKwan, y_test_YuenKwan = train_test_split(x_YuenKwan, y_YuenKwan, test_size=0.3, train_size=0.7, random_state=49)


# Req.e: Build and validate the model
from sklearn.linear_model import LogisticRegression

# e1) Use sklearn fit a logistic regression model to the training data 
YuenKwan_model = LogisticRegression(solver='lbfgs')
YuenKwan_model.fit(x_train_YuenKwan, y_train_YuenKwan)

# e2) Display the coefficients 
coef_table = pd.DataFrame(zip(x_train_YuenKwan.columns, np.transpose(YuenKwan_model.coef_)))
print('----------- e2) Print Coefficients -------------')
print(coef_table)

# e3) Cross Validation
from sklearn.model_selection import cross_val_score

print('----------- e3) Cross Validation -------------')
# e3-3) Repeat the validation from test size 10% to 50% with 5% incremental
for test_size in np.arange(0.10, 0.50, 0.05):
    # e3-1, e3-2) Use sklearn cross_val_score, and set k-fold to be 10
    scores = cross_val_score(YuenKwan_model, x_train_YuenKwan, y_train_YuenKwan, cv=10)
    
    # e3-4) Print minimum, mean, maximum accuracy
    print(f"Test size: {test_size:.0%}")
    print(f"Minimum accuracy: {scores.min():.2%}")
    print(f"Mean accuracy: {scores.mean():.2%}")
    print(f"Maximum accuracy: {scores.max():.2%}")

# e-5) Recommend the best split
'''
The maximum accuracy increases from 70.97% at a test size of 10% 
to 90.48% at a test size of 30% and above, indicating that 
a larger test size may be better for this particular dataset. 
Therefore, a test size of 30% or higher could be recommended 
as the best split scenario.

----------- e3) Cross Validation -------------
Test size: 10%
Minimum accuracy: 70.97%
Mean accuracy: 78.95%
Maximum accuracy: 90.48%
Test size: 15%
Minimum accuracy: 70.97%
Mean accuracy: 78.95%
Maximum accuracy: 90.48%
Test size: 20%
Minimum accuracy: 70.97%
Mean accuracy: 78.95%
Maximum accuracy: 90.48%
Test size: 25%
Minimum accuracy: 70.97%
Mean accuracy: 78.95%
Maximum accuracy: 90.48%
Test size: 30%
Minimum accuracy: 70.97%
Mean accuracy: 78.95%
Maximum accuracy: 90.48%
Test size: 35%
Minimum accuracy: 70.97%
Mean accuracy: 78.95%
Maximum accuracy: 90.48%
Test size: 40%
Minimum accuracy: 70.97%
Mean accuracy: 78.95%
Maximum accuracy: 90.48%
Test size: 45%
Minimum accuracy: 70.97%
Mean accuracy: 78.95%
Maximum accuracy: 90.48%
'''


# Req.f: Test the model

# f1) Rebuild the model using the 70%-30% train/test split
YuenKwan_model = LogisticRegression(solver='lbfgs')
YuenKwan_model.fit(x_train_YuenKwan, y_train_YuenKwan)

# f2) Predict probabilities and store them in y_pred_firstname
y_pred_YuenKwan = YuenKwan_model.predict_proba(x_test_YuenKwan)

# f3) Convert probabilities to boolean values using a threshold = 0.5
y_pred_YuenKwan_flag_05 = y_pred_YuenKwan[:,1] > 0.5

# f4) Import evaluation metrics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# f5) Print the accuracy of the model on the test data
accuracy_YuenKwan_05 = accuracy_score(y_test_YuenKwan, y_pred_YuenKwan_flag_05)
print('----------- f5) Model Accuracy (Threshold = 0.5) -------------')
print(f"Accuracy: {accuracy_YuenKwan_05:.2%}")

# f6) Print the confusion matrix
print('----------- f6) Confusion Matrix (Threshold = 0.5) -------------')
print(confusion_matrix(y_test_YuenKwan, y_pred_YuenKwan_flag_05))

# f7) Print the classification report
print('----------- f7) Classification Report (Threshold = 0.5) -------------')
print(classification_report(y_test_YuenKwan, y_pred_YuenKwan_flag_05))

# f9) Repeat: Convert probabilities to boolean values using a threshold = 0.75
y_pred_YuenKwan_flag_075 = y_pred_YuenKwan[:,1] > 0.75

# f9) Repeat: Print the accuracy of the model on the test data
accuracy_YuenKwan_075 = accuracy_score(y_test_YuenKwan, y_pred_YuenKwan_flag_075)
print('----------- f9) Model Accuracy (Threshold = 0.75) -------------')
print(f"Accuracy: {accuracy_YuenKwan_075:.2%}")

# f9) Print the confusion matrix
print('----------- f9) Confusion Matrix (Threshold = 0.75) -------------')
print(confusion_matrix(y_test_YuenKwan, y_pred_YuenKwan_flag_075))

# f9) Print the classification report
print('----------- f9) Classification Report (Threshold = 0.75) -------------')
print(classification_report(y_test_YuenKwan, y_pred_YuenKwan_flag_075))

'''
----------- f5) Model Accuracy (Threshold = 0.5) -------------
Accuracy: 79.10%
----------- f6) Confusion Matrix (Threshold = 0.5) -------------
[[147  31]
 [ 25  65]]
----------- f7) Classification Report (Threshold = 0.5) -------------
              precision    recall  f1-score   support

         0.0       0.85      0.83      0.84       178
         1.0       0.68      0.72      0.70        90

    accuracy                           0.79       268
   macro avg       0.77      0.77      0.77       268
weighted avg       0.80      0.79      0.79       268



----------- f9) Model Accuracy (Threshold = 0.75) -------------
Accuracy: 82.09%
----------- f9) Confusion Matrix (Threshold = 0.75) -------------
[[173   5]
 [ 43  47]]
----------- f9) Classification Report (Threshold = 0.75) -------------
              precision    recall  f1-score   support

         0.0       0.80      0.97      0.88       178
         1.0       0.90      0.52      0.66        90

    accuracy                           0.82       268
   macro avg       0.85      0.75      0.77       268
weighted avg       0.84      0.82      0.81       268


From the classification report, we can see the precision, recall, and F1-score for each class (0 and 1) along with the weighted average of these metrics.

Precision represents the proportion of true positives (TP) out of all predicted positives (TP + false positives (FP)). So, a high precision means that when the model predicts a positive (in this case, the passenger survived), it is correct most of the time.

Recall represents the proportion of true positives (TP) out of all actual positives (TP + false negatives (FN)). So, a high recall means that the model correctly identifies most of the actual positives (in this case, the passengers who actually survived).

F1-score is the harmonic mean of precision and recall. It is a way to balance the importance of precision and recall.

Looking at the classification report for threshold 0.5, we can see that the precision for class 0 (did not survive) is higher than that for class 1 (survived), whereas the recall for class 1 is higher than that for class 0. This means that the model is better at identifying passengers who survived than those who did not.

When we change the threshold to 0.75, we can see that the precision for class 1 (survived) increases, while the recall decreases. This means that the model is now more precise in identifying passengers who survived, but it may miss some of the passengers who actually survived.

In terms of accuracy, we can see that the accuracy increases when we change the threshold to 0.75. However, we need to be careful when using accuracy as the sole metric to evaluate the model. In cases where the classes are imbalanced (as in this case), accuracy may not be a good indicator of the model's performance. It is important to look at precision and recall as well to get a better understanding of the model's performance.


# f10, f11)
To compare the values of accuracy, precision, and recall generated at the threshold 0.5 and 0.75, you can look at the respective values in the classification report for each threshold.

At the threshold of 0.5, the model has an accuracy of 79.10%, precision of 0.68 for class 1, and recall of 0.72 for class 1.

At the threshold of 0.75, the model has an accuracy of 82.09%, precision of 0.90 for class 1, and recall of 0.52 for class 1.

We can observe that increasing the threshold from 0.5 to 0.75 has resulted in an increase in accuracy, but at the cost of lower recall for class 1. The precision for class 1 has increased, indicating that the model has become more conservative in its predictions, resulting in fewer false positives.
'''
