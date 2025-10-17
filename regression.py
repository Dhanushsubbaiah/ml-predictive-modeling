import time
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
# Load your dataset
# data = np.load('your_dataset.npy')
data = np.loadtxt("GasProperties.csv", delimiter=",",dtype=float)
# Assume that the last column is the target (Idx)
X = data[:, :-1] # features
y = data[:, -1] # target
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)
# Standardize the features
scaler = StandardScaler()
X_train_S = scaler.fit_transform(X_train)
X_test_S = scaler.transform(X_test)
# Initialize the models
linear_regression = LinearRegression()
print('Linear Regression\n')
# Fit and evaluate the models
for order in range(1, 15): # change the range as needed
 start_time = time.time()
 if order > 1:
 poly_features = PolynomialFeatures(degree=order, include_bias=False)
 X_train_poly = poly_features.fit_transform(X_train_S)
 X_test_poly = poly_features.transform(X_test_S)
 linear_regression.fit(X_train_poly, y_train)
 y_train_pred = linear_regression.predict(X_train_poly)
 y_test_pred = linear_regression.predict(X_test_poly)
 else:
 linear_regression.fit(X_train_S, y_train)
 y_train_pred = linear_regression.predict(X_train_S)
 y_test_pred = linear_regression.predict(X_test_S)
 training_time = time.time() - start_time

 # Calculate metrics
 train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
 train_r2 = r2_score(y_train, y_train_pred)
 test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
 test_r2 = r2_score(y_test, y_test_pred)
 time_delta = str(datetime.timedelta(seconds=training_time))
 print(f'Order: {order}, Train RMSE: {train_rmse}, Train R2: {train_r2},
Training Time: {time_delta}, Test RMSE: {test_rmse}, Test R2: {test_r2}')
scaler = MinMaxScaler()
X_train_MN = scaler.fit_transform(X_train)
X_test_MN = scaler.transform(X_test)
def stochastic_gradient_descent(X, y, lr=0.001, epochs=100):
 m, n = X.shape
 theta = np.zeros(n) # initialize weights
 history = np.zeros(epochs) # to store cost
 for epoch in range(epochs):
 for i in range(m):
 random_index = np.random.randint(m)
 xi = X[random_index:random_index+1]
 yi = y[random_index:random_index+1]
 gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
 theta = theta - lr * gradients
 history[epoch] = np.sqrt(mean_squared_error(y, X.dot(theta)))
 return theta, history
# Add a column of ones to X_train for the bias term
X_train_bias = np.c_[np.ones((X_train_MN.shape[0], 1)), X_train_MN]
print('\nGradient Descent method\n')
for order in range(1, 10):
 start_time = time.time()
 if order > 1:
 poly_features = PolynomialFeatures(degree=order, include_bias=False)
 X_train_poly = poly_features.fit_transform(X_train_bias)
 X_test_poly = poly_features.transform(np.c_[np.ones((X_test_MN.shape[0],
1)), X_test_MN])
 else:
 X_train_poly = X_train_bias
 X_test_poly = np.c_[np.ones((X_test_MN.shape[0], 1)), X_test_MN]
 theta, history = stochastic_gradient_descent(X_train_poly, y_train)
 training_time = time.time() - start_time
 # Predictions
 y_train_pred = X_train_poly.dot(theta)
 y_test_pred = X_test_poly.dot(theta)
 # Calculate metrics
 train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
 train_r2 = r2_score(y_train, y_train_pred)
 test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
 test_r2 = r2_score(y_test, y_test_pred)
 time_delta = str(datetime.timedelta(seconds=training_time))
 print(f'Order: {order}, Train RMSE: {train_rmse}, Train R2: {train_r2},
Training Time: {time_delta}, Test RMSE: {test_rmse}, Test R2: {test_r2}')
lasso_cv = LassoCV(cv=10,tol=0.1,max_iter=10000)
print('\nLASSO method\n')
# Fit and evaluate the models
for order in range(1, 10): # change the range as needed
 start_time = time.time()
 if order > 1:
 poly_features = PolynomialFeatures(degree=order, include_bias=False)
 X_train_poly = poly_features.fit_transform(X_train_S)
 X_test_poly = poly_features.transform(X_test_S)
 lasso_cv.fit(X_train_poly, y_train)
 y_train_pred = lasso_cv.predict(X_train_poly)
 y_test_pred = lasso_cv.predict(X_test_poly)
 else:
 lasso_cv.fit(X_train_S, y_train)
 y_train_pred = lasso_cv.predict(X_train_S)
 y_test_pred = lasso_cv.predict(X_test_S)
 training_time = time.time() - start_time
 # Convert seconds to a timedelta object
 time_delta = datetime.timedelta(seconds=training_time)
 # Format the timedelta object into a string of format "H:MM:SS.mmm"
 formatted_time = str(time_delta)
 # Calculate metrics
 train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
 train_r2 = r2_score(y_train, y_train_pred)
 test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
 test_r2 = r2_score(y_test, y_test_pred)
 training_time = time.time() - start_time
 print(f'Order: {order}, Train RMSE: {train_rmse}, Train R2: {train_r2},
Training Time: {formatted_time}, Test RMSE: {test_rmse}, Test R2: {test_r2},
Lambda: {lasso_cv.alpha_}')
