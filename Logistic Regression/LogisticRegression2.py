# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ad_data = pd.read_csv(
    'C:/Users/Faizi/Desktop/MachineLeariningUsingScikit-learn/Logistic Regression/advertising.csv')

ad_data.head()
ad_data.info()
ad_data.describe()

# data analysis
sns.set_style('whitegrid')
ad_data['Age'].hist(bins=30)
plt.xlabel('Age')

sns.jointplot(x='Age', y='Area Income', data=ad_data)
sns.jointplot(x='Age', y='Daily Time Spent on Site',
              data=ad_data, color='red', kind='kde')
sns.jointplot(x='Daily Time Spent on Site',
              y='Daily Internet Usage', data=ad_data, color='green')
sns.pairplot(ad_data, hue='Clicked on Ad', palette='bwr')

# logisitic regression model
from sklearn.model_selection import train_test_split
X = ad_data[['Daily Time Spent on Site', 'Age',
             'Area Income', 'Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

#predictions and evaluations
predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
