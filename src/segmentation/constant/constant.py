import os

# Schema File path
ROOT_DIR=os.getcwd()
CONFIG_DIR='config'
SCHEMA_FILE='schema.yaml'
SCHEMA_FILE_PATH=os.path.join(ROOT_DIR,CONFIG_DIR,SCHEMA_FILE)

# Config File path
ROOT_DIR=os.getcwd()
CONFIG_DIR='config'
SCHEMA_FILE='config.yaml'
CONFIG_FILE_PATH=os.path.join(ROOT_DIR,CONFIG_DIR,SCHEMA_FILE)

# Model Config Path 
# Config File path
ROOT_DIR=os.getcwd()
CONFIG_DIR='config'
MODEL_TRAINING_CONFIG='model_training.yaml'
MODEL_TRAINING_CONFIG_PATH=os.path.join(ROOT_DIR,CONFIG_DIR,MODEL_TRAINING_CONFIG)



# Data Ingestion 
# Data Ingestion Config
DATA_INGESTION_CONFIG_KEY = "data_ingestion_config"
DATA_INGESTION_DATABASE_NAME= "data_base"
DATA_INGESTION_COLLECTION_NAME= "collection_name"
DATA_INGESTION_ARTIFACT_DIR = "artifact_dir"
DATA_INGESTION_RAW_DATA_DIR_KEY = "raw_data_dir"
DATA_INGESTION_INGESTED_DIR_NAME_KEY = "ingested_dir"
DATA_INGESTION_TRAIN_DIR_KEY = "ingested_train_dir"

# Data Validation related variable
DATA_VALIDATION_ARTIFACT_DIR="data_validation_dir"
DATA_VALIDATION_CONFIG_KEY = "data_validation_config"
DATA_VALIDATION_SCHEMA_FILE_NAME_KEY = "schema_file_name"
DATA_VALIDATION_SCHEMA_DIR_KEY = "schema_dir"
DATA_VALIDATION_VALID_DATASET ="validated_data"
DATA_VALIDATION_TRAIN_FILE = "Train_data"

# transformation config file  
ROOT_DIR=os.getcwd()
CONFIG_DIR='config'
TRANSFORMATION_FILE='transformation.yaml'
TRANFORMATION_YAML_FILE_PATH=os.path.join(ROOT_DIR,CONFIG_DIR,TRANSFORMATION_FILE)
TARGET_COLUMN_KEY= 'target_column'
NUMERICAL_COLUMN_KEY= 'numerical_columns'
CATEGORICAL_COLUMNS ='encode_columns'
DROP_COLUMNS= 'drop_columns'


# key  ---> config.yaml---->values
DATA_TRANSFORMATION_CONFIG_KEY = "data_transformation_config"
DATA_TRANSFORMATION='data_transformation_dir'
DATA_TRANSFORMATION_DIR_NAME_KEY = "transformed_dir"
DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY = "transformed_train_dir"
DATA_TRANSFORMATION_TEST_DIR_NAME_KEY = "transformed_test_dir"
DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY = "preprocessing_dir"
DATA_TRANSFORMATION_PREPROCESSING_FILE_NAME_KEY = "preprocessed_object_file_name"
DATA_TRANSFORMATION_FEA_ENG_FILE_NAME_KEY='feature_eng_file'
DATA_TRANSFORMATION_PREPROCESSOR_NAME_KEY='preprocessed_object_file_name'

# Prediction Yaml file path 
ROOT_DIR=os.getcwd()
CONFIG_DIR='config'
PREDICTION_YAML_FILE='prediction.yaml'
PREDICTION_YAML_FILE_PATH=os.path.join(ROOT_DIR,CONFIG_DIR,PREDICTION_YAML_FILE)

# Model Training 
MODEL_TRAINING_CONFIG_KEY='model_trainer_config'
MODEL_TRAINER_ARTIFACT_DIR = "model_training"
MODEL_TRAINER_OBJECT = "model_object"
MODEL_REPORT_FILE="model_report"


# Param Optimisation 
PARAM_OPTIMIZE_CONFIG_KEY='param_optimize_config'
PARAM_OPTIMIZE_DIRECTORY='param_optimize_dir'
PARAM_OPTIMIZE_MODEL='model_object'
PARAM_OPTIMIZE_MODEL_REPORT='model_report'

# Saved Model 
SAVED_MODEL_CONFIG_KEY='saved_model_config'
SAVED_MODEL_DIR='directory'
SAVED_MODEL_OBJECT='model_object'
SAVED_MODEL_REPORT='model_report'

##  model evaluation 
MODEL_EVAL_CONFIG_KEY='model_eval_config'
MODEL_EVALUATION_DIRECTORY='model_eval_dir'
MODEL_EVALUATION_OBJECT='model_object'
MODEL_REPORT='model_report'
MODEL_LABEL='model_name'



AWS_CONFIG_KEY ='Aws_config'
# Bucket Details
S3_BUCKET='S3_bucket'
BUCKET_NAME='bucket_name'



from segmentation.utils.main_utils import read_yaml_file


# Model Training Parameters 
MODEL_TRAINING_YAML = os.path.join(os.getcwd(),'config','model_training.yaml')
training_config_data=read_yaml_file(MODEL_TRAINING_YAML)
MODEL_NAME=training_config_data['model']['Model_name']
METRIC=training_config_data['model']['model_metric']
parameters=training_config_data['model']['parameters']
CLUSTERS=parameters['no_of_clusters']
COVARIANCE=parameters['covariance_type']
INIT_PARAMS=parameters['init_params']
ITERATIONS=int(parameters['max_iter'])
TOLERANCE=float(parameters['tol'])






TRANSFORMATION_YAML_FILE_PATH=os.path.join(os.getcwd(),'config','transformation.yaml')

## Artifact Entity 
file_path=os.path.join(ROOT_DIR,'src','segmentation','entity','artifact_entity.yaml')
ARTIFACT_ENTITY_YAML_FILE_PATH=file_path

