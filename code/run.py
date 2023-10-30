from analyse_data import load_data, explore_data, remove_features, handle_missing_data
from analyse_data import analyze_feature, identify_outliers_iqr, analyze_feature_relationships, transform_categorical_feature
from linear_regression import split_data, train_and_evaluate_model, feature_selection
from linear_regression import standardize_features, train_and_evaluate_standardized_model, visualize_correlation_matrix
from linear_regression import create_polynomial_features, train_and_evaluate_linear_regression
from linear_regression import train_and_evaluate_ridge, train_and_evaluate_lasso, visualize_coefficients
from knn import knn_classification, find_best_knn_params

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
    
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(df)


    print('Train and evaluate the model')
    regression_model = train_and_evaluate_model(x_train, y_train, x_val, y_val, x_test, y_test)


    feature_selection(x_train, y_train)


    x_train_std, x_val_std, x_test_std = standardize_features(x_train, x_val, x_test)


    print('Train and evaluate the standardized model')
    regression_model_std = train_and_evaluate_standardized_model(x_train_std, y_train, x_val_std, y_val, x_test_std, y_test, x_train)


    visualize_correlation_matrix(x_train)
    
    
    print('Create polynomial features and evaluate')
    x_train_poly, x_val_poly, x_test_poly = create_polynomial_features(x_train_std, x_val_std, x_test_std, degree=2)
    regression_model_poly = train_and_evaluate_linear_regression(x_train_poly, y_train, x_val_poly, y_val, x_test_poly, y_test)


    print('Train and evaluate Ridge regression:')
    ridge_model = train_and_evaluate_ridge(x_train, y_train, x_val, y_val, x_test, y_test, alpha=5)


    print('Train and evaluate Lasso regression:')
    lasso_model = train_and_evaluate_lasso(x_train, y_train, x_val, y_val, x_test, y_test, alpha=0.01)


    visualize_coefficients(regression_model_poly, ridge_model, lasso_model)


    best_params = find_best_knn_params(x_train, y_train)
    knn_classification(x_train, y_train, x_test, y_test, best_params)


if __name__ == "__main__":
    main()