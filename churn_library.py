"""
Module to find customers who are likely to churn
Author: Omar Zeb
Date: June 25 2022
"""

# import libraries
import os
import logging
import pandas as pd
import numpy as np
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logging.basicConfig(
    filename=os.path.join(
        os. getcwd(),
        "logs",
        "churn_library.log"),
    filemode="w",
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s')


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    try:
        logging.info("Reading csv {}".format(pth))
        assert isinstance(pth, str)
        logging.info("Path is correct")

        df = pd.read_csv(pth)
        logging.info("SUCESS: csv read correctly")
        return df

    except AssertionError:
        logging.error("ERROR: path is not string")

    except FileNotFoundError:
        logging.error("ERROR: csv not found in path {}".format(pth))


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''

    # print first five rows of the dataframe
    print("First five rows of data are: ", df.head())

    print("The shape of the data is:", df.shape)

    print(
        "Total number of null entries per column in the data are:",
        df.isnull().sum())

    print("Data Summary is:", df.describe())

    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    image_dir = os.path.join(os.getcwd(), "images")
    # Plot the histogram of Customer Churn
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.title("Distribution of Churn")
    plt.savefig(os.path.join(image_dir, "Churn_Hist.png"))
    plt.close()

    # Plot the histogram of Customer Age
    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.title("Age Distribution of Customers")
    plt.savefig(os.path.join(image_dir, "Age_Hist.png"))
    plt.close()

    # Plot the histogram of Customer Marital Status
    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.title("Matital Distribution of Customers")
    plt.savefig(os.path.join(image_dir, "Marital_Hist.png"))
    plt.close()

    # Plot The transaction distribtion
    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.title(" Total Trans Ct distribution")
    plt.savefig(os.path.join(image_dir, "Trans_CT.png"))
    plt.close()

    # Plot the HeatMap of the data
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.title("Heatmap")
    plt.savefig(os.path.join(image_dir, "HeatMap.png"))
    plt.close()


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name
                      [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''

    for category in category_lst:
        out_list = []
        groups = df.groupby(category).mean()[response]

        for val in df[category]:
            out_list.append(groups.loc[val])

        df["{}_{}".format(category, response)] = out_list

    return df


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name
                       [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    y = df['Churn']
    X = pd.DataFrame()

    up_df = encoder_helper(df, cat_columns, response)

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    X[keep_cols] = up_df[keep_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    image_dir = os.path.join(os.getcwd(), "images")

    # Plot the results of logistic Regression
    plt.figure()
    plt.rc('figure', figsize=(20, 10))

    plt.text(0.01, 1.25, str('Logistic Regression Training Results'))

    plt.text(0.01, 0.05, str(classification_report(y_test,
                                                   y_test_preds_lr)))

    plt.text(0.01, 0.6, str('Logistic Regression Testing Results'))
    plt.text(0.01, 0.7, str(classification_report(y_train,
                                                  y_train_preds_lr)))

    plt.axis('off')
    plt.savefig(os.path.join(image_dir, 'logistic_regression_results.png'))
    plt.close()

    # Plot the results of random forest classifier
    plt.figure()
    plt.rc('figure', figsize=(20, 10))

    plt.text(0.01, 1.25, str('Random Forest Training Results'))

    plt.text(0.01, 0.05, str(classification_report(y_test,
                                                   y_test_preds_rf)))

    plt.text(0.01, 0.6, str('Random Forest Testing Results'))
    plt.text(0.01, 0.7, str(classification_report(y_train,
                                                  y_train_preds_rf)))

    plt.axis('off')
    plt.savefig(os.path.join(image_dir, 'random_forest_results.png'))
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # Save the figure
    plt.savefig(output_pth)
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # Random Forest
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    # LogisticRegression
    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # save the roc curve with score
    image_dir = os.path.join(os.getcwd(), "images")

    lrc_plot = plot_roc_curve(lrc, X_test, y_test)

    plt.figure(figsize=(15, 8))
    axis = plt.gca()

    _ = plot_roc_curve(cv_rfc.best_estimator_,
                       X_test, y_test, ax=axis, alpha=0.8)

    lrc_plot.plot(ax=axis, alpha=0.8)
    plt.savefig(os.path.join(image_dir, 'roc_curve.png'))
    plt.close()

    logging.info("Save best model.")

    model_dir = os.path.join(os.getcwd(), "model")
    joblib.dump(
        cv_rfc.best_estimator_,
        os.path.join(
            model_dir,
            'rfc_model.pkl'))
    joblib.dump(lrc, os.path.join(model_dir, 'logistic_model.pkl'))

    # store model results
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)

    # store feature importances plot
    feature_importance_plot(cv_rfc.best_estimator_, X_train,
                            os.path.join(image_dir, 'feature_importances.png'))


if __name__ == "__main__":
    logging.info("Loading Data")
    data = import_data("./data/BankChurners.csv")

    logging.info("Perform EDA.")
    perform_eda(data)

    logging.info("Split data into train & test")
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
        data, 'Churn')

    logging.info("Model training and Result Storing")
    train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
