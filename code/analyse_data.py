import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import kurtosis, skew
from fitter import Fitter

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def explore_data(df):
    print(df.head())
    num_features = df.shape[1]
    num_samples = df.shape[0]
    print(f"Number of features: {num_features}")
    print(f"Number of samples: {num_samples}")
    features = df.columns
    print(f"Features: {features}")
    missing_data = df.isnull().values.any()
    if missing_data:
        print("Missing data exists.")
        missing_count = df.isnull().sum()
        print(f"Missing data counts:\n{missing_count}")
    else:
        print("No missing data found.")
    return df

def remove_features(df, features_to_remove):
    df = df.drop(columns=features_to_remove)
    return df

def handle_missing_data(df, column, strategy='drop'):
    if strategy == 'drop':
        df.dropna(subset=[column], inplace=True)
    elif strategy == 'impute':
        pass
    return df

def identify_outliers_iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = series[((series < lower_bound) | (series > upper_bound))]
    return outliers

def analyze_feature(df, feature_name):
    print(df[feature_name].describe())
    sb.displot(data=df, x=feature_name, kind="hist", bins=5, aspect=1.5)
    height = df[feature_name].values
    print('Skewness: %.2f' % skew(df[feature_name]))
    print('Kurtosis: %.2f' % kurtosis(df[feature_name]))
    f = Fitter(height, distributions=["norm"])
    f.fit()
    f.summary()
    plt.xlabel(feature_name)
    plt.ylabel('Probability')
    plt.legend(loc='upper left')
    plt.show()
    plt.boxplot(df[feature_name])
    plt.xlabel(feature_name)
    plt.grid()

def analyze_feature_relationships(df, target_feature):
    correlation_matrix = df.corr()
    correlations_with_target = correlation_matrix[target_feature]
    print("Correlations with the target feature:")
    print(correlations_with_target)

def transform_categorical_feature(df, feature_name):
    df_dummy = pd.get_dummies(df[feature_name], prefix=feature_name)
    df = pd.concat([df, df_dummy], axis=1)
    df.drop([feature_name], axis=1, inplace=True)
    return df

def main():
    file_path = 'GuangzhouPM20100101_20151231.csv'
    df = load_data(file_path)
    df = explore_data(df)
    features_to_remove = ['No', 'PM_City Station', 'PM_5th Middle School']
    df = remove_features(df, features_to_remove)
    missing_data_columns = ['season', 'PM_US Post']
    for column in missing_data_columns:
        df = handle_missing_data(df, column)
    features_with_outliers = ['HUMI', 'DEWP', 'PRES', 'Iws', 'precipitation', 'Iprec']
    for feature_name in features_with_outliers:
        outliers = identify_outliers_iqr(df[feature_name])
        print(f"Outliers in {feature_name}:\n{outliers}")
    analyze_feature(df, 'TEMP')
    analyze_feature(df, 'DEWP')
    analyze_feature(df, 'HUMI')
    analyze_feature(df, 'PRES')
    analyze_feature(df, 'Iws')
    analyze_feature(df, 'precipitation')
    analyze_feature(df, 'Iprec')
    df = transform_categorical_feature(df, 'cbwd')
    analyze_feature(df, 'PM_US Post')
    analyze_feature_relationships(df, 'PM_US Post')

if __name__ == "__main":
    main()
