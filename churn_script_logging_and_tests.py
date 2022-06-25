import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(import_data):
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = import_data("./data/BankChurners.csv")
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda(perform_eda):
	'''
	test perform eda function
	'''

	perform_eda(cls.import_data("./data/BankChurners.csv"))

	image_dir = os.path.join(os.getcwd(), "images", "eda")

	images = os.listdir(image_dir)

	image_names = ["Age_Hist", "Churn_Hist", "HeatMap", "Marital_Hist", "Trans_CT"]
	for name in image_names:
		try:
			assert images.count("{}.png".format(name))
			logging.info("SUCESS: {}.png image present in eda folder".format(name))
		
		except:
			logging.error("ERROR: {}.png image not present in eda folder".format(name))



def test_encoder_helper(encoder_helper):
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

	df = encoder_helper(cls.import_data("./data/BankChurners.csv"), cat_columns, 'Churn')

	for col in cat_columns:
		try:
			assert col in df.columns
			logging.info("SUCESS: {} is present in the data".format(col))
		
		except:
			logging.error("ERROR: {} not present in the data".format(col))


def test_perform_feature_engineering(perform_feature_engineering):
	'''
	test perform_feature_engineering
	'''
	X_train, X_test, y_train, y_test = perform_feature_engineering(
        cls.import_data("./data/BankChurners.csv"), 'Churn')

	try:
		assert X_train[0].shape > 0
		assert X_train[1].shape > 0
		logging.info("SUCESS: X train has the right shape")
	
	except:
		logging.error("ERROR: X train does not have the right shape")

	try:
		assert X_test[0].shape > 0
		assert X_test[1].shape > 0
		logging.info("SUCESS: X test have the right shape")
	
	except:
		logging.error("ERROR: X test does not have the right shape")

	try:
		assert len(y_train) > 0
		logging.info("SUCESS: y train contains data")
	
	except:
		logging.error("ERROR: y train does not have any data")

	try:
		assert len(y_test) > 0
		logging.info("SUCESS: y test contains data")
	
	except:
		logging.error("ERROR: y test does not have any data")


def test_train_models(train_models):
	'''
	test train_models
	'''
	X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
        cls.import_data("./data/BankChurners.csv"), 'Churn')

	train_models(X_train, X_test, y_train, y_test)

	image_names = ["feature_importances", "logistic_regression_results", "random_forest_results", "roc_curve"]

	image_dir = os.path.join(os.getcwd(), "images", "results")
	images = os.listdir(image_dir)
	for name in image_names:
		try:
			assert images.count("{}.png".format(name))
			logging.info("SUCESS: {}.png is present in results folder".format(name))
		
		except:
			logging.error('ERROR: {}.png is not present in results folder'.format(name))

	model_names = ["logistic_model", "rfc_model"]

	model_dir = os.path.join(os.getcwd(), "model")
	models = os.listdir(model_dir)
	
	for name in model_names:
		try:
			assert models.count("{}.pkl".format(name))
			logging.info("SUCESS: {}.pkl model is present in model folder".format(name))
		
		except:
			logging.error('ERROR: {}.pkl is not present in model folder'.format(name))



if __name__ == "__main__":
	test_import(cls.import_data)

	test_eda(cls.perform_eda)

	test_encoder_helper(cls.encoder_helper)

	test_perform_feature_engineering(cls.perform_feature_engineering)
	
	test_train_models(cls.train_models)








