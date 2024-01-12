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
            self.test_file_path = os.path.join(self.ingested_data_dir,data_ingestion_key[DATA_INGESTION_TEST_DIR_KEY])
            self.split_size = data_ingestion_key[SPLIT_SIZE]
        except Exception  as e:
            raise ApplicationException(e,sys)    

            
            