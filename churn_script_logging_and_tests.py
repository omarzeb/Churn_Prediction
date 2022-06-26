"""
Performes tests for churn_library module
Author: Omar Zeb
Date: 26 June 2022
"""

import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err

    return df


def test_eda(perform_eda, df):
    '''
    test perform eda function
    '''
    perform_eda(df)

    image_dir = os.path.join(os.getcwd(), "images", "eda")

    images = os.listdir(image_dir)

    image_names = [
        "churn_distribution",
        "customer_age_distribution",
        "marital_status_distribution",
        "total_trans_Ct",
        "heatmap"]
    for img_name in image_names:
        try:
            assert images.count("{}.png".format(img_name))
            logging.info(
                "SUCESS: %s.png image present in eda folder", img_name)

        except AssertionError as err:
            logging.error(
                "ERROR: %s.png image not present in eda folder", img_name)

            raise err


def test_encoder_helper(encoder_helper, data):
    '''
    test encoder helper
    '''
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    df = encoder_helper(
        data,
        cat_columns,
        'Churn')

    for col in cat_columns:
        try:
            assert col in df.columns
            logging.info("SUCESS: %s is present in the data", col)

        except AssertionError as err:
            logging.error("ERROR: %s not present in the data", col)
            raise err

    return df


def test_perform_feature_engineering(perform_feature_engineering, df):
    '''
    test perform_feature_engineering
    '''

    x_train, x_test, y_train, y_test = perform_feature_engineering(
        df, 'Churn')

    try:
        assert x_train.shape[0] > 0
        assert x_train.shape[1] > 0
        logging.info("SUCESS: X train has the right shape")

    except AssertionError as err:
        logging.error("ERROR: X train does not have the right shape")
        raise err

    try:
        assert x_test.shape[0] > 0
        assert x_test.shape[1] > 0
        logging.info("SUCESS: X test have the right shape")

    except AssertionError as err:
        logging.error("ERROR: X test does not have the right shape")
        raise err

    try:
        assert len(y_train) > 0
        logging.info("SUCESS: y train contains data")

    except AssertionError as err:
        logging.error("ERROR: y train does not have any data")
        raise err

    try:
        assert len(y_test) > 0
        logging.info("SUCESS: y test contains data")

    except AssertionError as err:
        logging.error("ERROR: y test does not have any data")
        raise err

    return x_train, x_test, y_train, y_test


def test_train_models(train_models, x_train, x_test, y_train, y_test):
    '''
    test train_models
    '''
    train_models(x_train, x_test, y_train, y_test)

    image_names = [
        "feature_importances",
        "logistic_results",
        "rf_results",
        "roc_curve"]

    image_dir = os.path.join(os.getcwd(), "images", "results")
    images = os.listdir(image_dir)
    for name in image_names:
        try:
            assert images.count("{}.png".format(name))
            logging.info(
                "SUCESS: %s.png is present in results folder", name)

        except AssertionError as err:
            logging.error(
                'ERROR: %s.png is not present in results folder', name)
            raise err

    model_names = ["logistic_model", "rfc_model"]

    model_dir = os.path.join(os.getcwd(), "model")
    models = os.listdir(model_dir)

    for name in model_names:
        try:
            assert models.count("{}.pkl".format(name))
            logging.info(
                "SUCESS: %s.pkl model is present in model folder", name)

        except AssertionError as err:
            logging.error(
                'ERROR: %s.pkl is not present in model folder', name)
            raise err


if __name__ == "__main__":
    data_df = test_import(cls.import_data)

    test_eda(cls.perform_eda, data_df)

    data_df = test_encoder_helper(cls.encoder_helper, data_df)

    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = test_perform_feature_engineering(
        cls.perform_feature_engineering, data_df)

    test_train_models(cls.train_models, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
