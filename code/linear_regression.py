import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

def split_data(data):
    x = data.drop(['PM_US Post'], axis=1).copy()
    y = data['PM_US Post'].copy()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=data['month'])
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

    # Dropping specific rows if needed
    x_test.drop(47939, axis='index', inplace=True)
    y_test.drop(47939, axis='index', inplace=True)

    return x_train, y_train, x_val, y_val, x_test, y_test

def model_evaluation(y_true, y_predicted, N, d):
    mse = mean_squared_error(y_true, y_predicted)
    mae = mean_absolute_error(y_true, y_predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_predicted)
    r2_adj = 1 - (1 - r2) * (N - 1) / (N - d - 1)

    print('Mean squared error:', mse)
    print('Mean absolute error:', mae)
    print('Root mean squared error:', rmse)
    print('R2 score:', r2)
    print('R2 adjusted score:', r2_adj)

def train_and_evaluate_model(x_train, y_train, x_val, y_val, x_test, y_test):
    regression_model = LinearRegression(fit_intercept=True)
    regression_model.fit(x_train, y_train)

    y_predicted_val = regression_model.predict(x_val)
    model_evaluation(y_val, y_predicted_val, x_train.shape[0], x_train.shape[1])

    regression_model.fit(pd.concat([x_train, x_val], ignore_index=True), pd.concat([y_train, y_val], ignore_index=True))

    y_predicted_test = regression_model.predict(x_test)
    model_evaluation(y_test, y_predicted_test, x_train.shape[0], x_train.shape[1])

    return regression_model

def feature_selection(x_train, y_train):
    X = sm.add_constant(x_train)
    model = sm.OLS(y_train, X.astype('float')).fit()
    print(model.summary())

    X = sm.add_constant(x_train.drop('HUMI', axis=1))
    model = sm.OLS(y_train, X.astype('float')).fit()
    print(model.summary())

    X = sm.add_constant(x_train.drop(['HUMI', 'DEWP'], axis=1))
    model = sm.OLS(y_train, X.astype('float')).fit()
    print(model.summary())

    X = sm.add_constant(x_train.drop(['HUMI', 'TEMP', 'DEWP'], axis=1))
    model = sm.OLS(y_train, X.astype('float')).fit()
    print(model.summary())

def standardize_features(x_train, x_val, x_test):
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_std = scaler.transform(x_train)
    x_val_std = scaler.transform(x_val)
    x_test_std = scaler.transform(x_test)
    return x_train_std, x_val_std, x_test_std

def train_and_evaluate_standardized_model(x_train_std, y_train, x_val_std, y_val, x_test_std, y_test, x_train):
    regression_model_std = LinearRegression()
    regression_model_std.fit(x_train_std, y_train)
    y_predicted = regression_model_std.predict(x_val_std)
    model_evaluation(y_val, y_predicted, x_train_std.shape[0], x_train_std.shape[1])

    # Convert numpy arrays to Pandas DataFrames
    x_train_std_df = pd.DataFrame(x_train_std, columns=x_train.columns)
    x_val_std_df = pd.DataFrame(x_val_std, columns=x_train.columns)

    regression_model_std.fit(pd.concat([x_train_std_df, x_val_std_df], ignore_index=True), pd.concat([y_train, y_val], ignore_index=True))
    y_predicted = regression_model_std.predict(x_test_std)
    model_evaluation(y_test, y_predicted, x_train_std.shape[0], x_train_std.shape[1])

    return regression_model_std

def visualize_correlation_matrix(data):
    corr_mat = data.corr()
    plt.figure(figsize=(12, 8), linewidth=2)
    sb.heatmap(corr_mat, annot=True)
    plt.show()

def create_polynomial_features(x_train_std, x_val_std, x_test_std, degree=1):
    poly = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=False)
    x_train_poly = poly.fit_transform(x_train_std)
    x_val_poly = poly.transform(x_val_std)
    x_test_poly = poly.transform(x_test_std)
    return x_train_poly, x_val_poly, x_test_poly

def train_and_evaluate_linear_regression(x_train, y_train, x_val, y_val, x_test, y_test):
    regression_model = LinearRegression()
    regression_model.fit(x_train, y_train)

    y_predicted_val = regression_model.predict(x_val)
    model_evaluation(y_val, y_predicted_val, x_train.shape[0], x_train.shape[1])

    # Convert NumPy arrays to DataFrames
    x_train_df = pd.DataFrame(x_train, columns=[f'feature_{i}' for i in range(x_train.shape[1])])
    x_val_df = pd.DataFrame(x_val, columns=[f'feature_{i}' for i in range(x_val.shape[1])])

    regression_model.fit(pd.concat([x_train_df, x_val_df], ignore_index=True), pd.concat([y_train, y_val], ignore_index=True))

    y_predicted_test = regression_model.predict(x_test)
    model_evaluation(y_test, y_predicted_test, x_train.shape[0], x_train.shape[1])

    return regression_model

def train_and_evaluate_ridge(x_train, y_train, x_val, y_val, x_test, y_test, alpha=1):
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(x_train, y_train)

    y_predicted_val = ridge_model.predict(x_val)
    model_evaluation(y_val, y_predicted_val, x_train.shape[0], x_train.shape[1])

    ridge_model.fit(pd.concat([x_train, x_val], ignore_index=True), pd.concat([y_train, y_val], ignore_index=True))

    y_predicted_test = ridge_model.predict(x_test)
    model_evaluation(y_test, y_predicted_test, x_train.shape[0], x_train.shape[1])

    return ridge_model

def train_and_evaluate_lasso(x_train, y_train, x_val, y_val, x_test, y_test, alpha=0.01):
    lasso_model = Lasso(alpha=alpha)
    lasso_model.fit(x_train, y_train)

    y_predicted_val = lasso_model.predict(x_val)
    model_evaluation(y_val, y_predicted_val, x_train.shape[0], x_train.shape[1])

    lasso_model.fit(pd.concat([x_train, x_val], ignore_index=True), pd.concat([y_train, y_val], ignore_index=True))

    y_predicted_test = lasso_model.predict(x_test)
    model_evaluation(y_test, y_predicted_test, x_train.shape[0], x_train.shape[1])

    return lasso_model

def visualize_coefficients(regression_model, ridge_model, lasso_model):
    plt.figure(figsize=(10, 5))
    plt.plot(regression_model.coef_, alpha=0.7, linestyle='none', marker='*', markersize=5, color='red', label='linear', zorder=7)
    plt.plot(ridge_model.coef_, alpha=0.5, linestyle='none', marker='d', markersize=6, color='blue', label='Ridge')
    plt.plot(lasso_model.coef_, alpha=0.4, linestyle='none', marker='o', markersize=7, color='green', label='Lasso')
    plt.xlabel('Coefficient Index', fontsize=16)
    plt.ylabel('Coefficient Magnitude', fontsize=16)
    plt.legend(fontsize=13, loc='best')
    plt.show()
