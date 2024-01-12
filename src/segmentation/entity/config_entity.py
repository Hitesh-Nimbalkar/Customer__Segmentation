import os,sys
from segmentation.exception.exception import ApplicationException
from segmentation.logger.logger import logging
from segmentation.utils.main_utils import read_yaml_file
from segmentation.constant import *



config_data=read_yaml_file(CONFIG_FILE_PATH)

class TrainingPipelineConfig:
    
    def __init__(self):
        try:
            training_pipeline_config=config_data['training_pipeline_config']
            artifact_dir=training_pipeline_config['artifact']
            self.artifact_dir = os.path.join(os.getcwd(),artifact_dir)
        except Exception  as e:
            raise ApplicationException(e,sys)    


class DataIngestionConfig:
    
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try:
            data_ingestion_key=config_data[DATA_INGESTION_CONFIG_KEY]
            
            self.database_name=data_ingestion_key[DATA_INGESTION_DATABASE_NAME]
            self.collection_name=data_ingestion_key[DATA_INGESTION_COLLECTION_NAME]
            
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir ,data_ingestion_key[DATA_INGESTION_ARTIFACT_DIR])
            self.raw_data_dir = os.path.join(self.data_ingestion_dir,data_ingestion_key[DATA_INGESTION_RAW_DATA_DIR_KEY])
            self.ingested_data_dir=os.path.join(self.raw_data_dir,data_ingestion_key[DATA_INGESTION_INGESTED_DIR_NAME_KEY])
            self.train_file_path = os.path.join(self.ingested_data_dir,data_ingestion_key[DATA_INGESTION_TRAIN_DIR_KEY])
        except Exception  as e:
            raise ApplicationException(e,sys)    

            

class DataValidationConfig:
    
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try:
        
            data_validation_key=config_data[DATA_VALIDATION_CONFIG_KEY]
            self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir ,data_validation_key[DATA_VALIDATION_ARTIFACT_DIR])
            self.validated_dir=os.path.join(training_pipeline_config.artifact_dir,data_validation_key[DATA_VALIDATION_VALID_DATASET])
            self.validated_train_path=os.path.join(self.data_validation_dir,data_validation_key[DATA_VALIDATION_TRAIN_FILE])
            self.schema_file_path=SCHEMA_FILE_PATH
        
        except Exception  as e:
            raise ApplicationException(e,sys)      

class DataTransformationConfig:
    
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        
        
        data_transformation_key=config_data[DATA_TRANSFORMATION_CONFIG_KEY]
        
        self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir , data_transformation_key[DATA_TRANSFORMATION])
        self.transformation_dir = os.path.join(self.data_transformation_dir,data_transformation_key[DATA_TRANSFORMATION_DIR_NAME_KEY])
        self.transformed_train_dir = os.path.join(self.transformation_dir,data_transformation_key[DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY])
        self.preprocessed_dir = os.path.join(self.data_transformation_dir,data_transformation_key[DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY])
        self.feature_engineering_object_file_path =os.path.join(self.preprocessed_dir,data_transformation_key[DATA_TRANSFORMATION_FEA_ENG_FILE_NAME_KEY])
        self.preprocessor_file_object_file_path=os.path.join(self.preprocessed_dir,data_transformation_key[DATA_TRANSFORMATION_PREPROCESSOR_NAME_KEY])


class ModelTrainerConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try:

            self.model_trainer_config = config_data[MODEL_TRAINING_CONFIG_KEY]
         
            self.trained_model_directory=os.path.join(training_pipeline_config.artifact_dir,
                                                    self.model_trainer_config[MODEL_TRAINER_ARTIFACT_DIR])

            self.trained_model_file_path = os.path.join(self.trained_model_directory,
                                                    self.model_trainer_config[MODEL_TRAINER_OBJECT])
            
            
            self.png_location=os.path.join(self.trained_model_directory)
            self.model_report_path=os.path.join(self.trained_model_directory,
                                        self.model_trainer_config[MODEL_REPORT_FILE])


        except Exception as e:
            raise ApplicationException(e,sys) from e