# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

customers = pd.read_csv(
    'C:/Users/Faizi/Desktop/MachineLeariningUsingScikit-learn/Linear Regression/Ecommerce Customers.csv')

customers.head()
customers.info()
customers.describe()

sns.set_palette("GnBu_d")
sns.set_style('whitegrid')
sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=customers)
sns.jointplot(x='Time on App',
              y='Yearly Amount Spent', data=customers)

sns.jointplot(x='Yearly Amount Spent', y = 'Length of Membership', kind='hex', data=customers)
sns.pairplot(customers)
sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=customers)

y = customers['Yearly Amount Spent']
X = customers[['Avg. Session Length', 'Time on App',
               'Time on Website', 'Length of Membership']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

print('Coefficients: \n', lm.coef_)

predictions = lm.predict(X_test)

plt.scatter(y_test, predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

sns.distplot((y_test-predictions), bins=50)

coeffecients = pd.DataFrame(lm.coef_, X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients
