# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

USAhousing = pd.read_csv('USA_Housing.csv')

# print(USAhousing.head())
# print(USAhousing.info())
# print(USAhousing.describe())
# USAhousing.columns

# sns.pairplot(USAhousing)

# sns.distplot(USAhousing['Price'])
sns.heatmap(USAhousing.corr())
plt.show()

X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                'Avg. Area Number of Bedrooms', 'Area Population']]

y = USAhousing['Price']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=101)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
print(lm.intercept_)

coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
# coeff_df

predictions = lm.predict(X_test)
# prediction

plt.scatter(y_test, predictions)
sns.distplot((y_test-predictions), bins=50)

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
