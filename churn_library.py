"""
Module to find customers who are likely to churn
Author: Omar Zeb
Date: June 25 2022
"""

# import libraries
import os
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

os.environ['QT_QPA_PLATFORM']='offscreen'

logging.basicConfig(filename=os.path.join(os. getcwd(), "logs", "churn_library.log"),
                        filemode="w",
                        level = logging.INFO,
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
        logging.Info("Reading csv {}".format(pth))
        assert isinstance(pth, str)
        logging.Info("Path is correct")

        df = pd.read_csv(pth)
        logging.Info("SUCESS: csv read correctly")
        return df

    except AssertionError:
        logging.ERROR("ERROR: path is not string")

    except FileNotFoundError:
        logging.ERROR("ERROR: csv not found in path {}".format(pth))


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    try:
        assert isinstance(df, pd.DataFrame)
        logging.Info("Input dataframe is correct")

        #print first five rows of the dataframe
        print("First five rows of data are: ",df.head())

        print("The shape of the data is:", df.shape())

        print("Total number of null entries per column in the data are:", df.isnull().sum())

        print("Data Summary is:", df.describe())

        df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)

        image_dir = os.path.join(os.cwd, "images")
        # Plot the histogram of Customer Churn
        plt.figure(figsize=(20,10)) 
        df['Churn'].hist()
        plt.title("Distribution of Churn")
        plt.savefig(os.path.join(image_dir, "Churn_Hist.png"))
        plt.close()

        # Plot the histogram of Customer Age
        plt.figure(figsize=(20,10)) 
        df['Customer_Age'].hist()
        plt.title("Age Distribution of Customers")
        plt.savefig(os.path.join(image_dir, "Age_Hist.png"))
        plt.close()

        # Plot the histogram of Customer Marital Status
        plt.figure(figsize=(20,10)) 
        df.Marital_Status.value_counts('normalize').plot(kind='bar')
        plt.title("Matital Distribution of Customers")
        plt.savefig(os.path.join(image_dir, "Marital_Hist.png"))
        plt.close()

        # Plot The transaction distribtion
        plt.figure(figsize=(20,10)) 
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


    except:
        logging.ERROR("ERROR: Dataframe not found")


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    
    for category in category_lst:
        out_list = []
        groups = df.groupby(category).mean()[response]

        for val in df['Gender']:
            out_list.append(groups.loc[val])
        
        df["{}_{}".format(category,response)] = out_list

    return df


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

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


    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn']

    X[keep_cols] = up_df[keep_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)

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
    pass


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
    pass

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
    pass