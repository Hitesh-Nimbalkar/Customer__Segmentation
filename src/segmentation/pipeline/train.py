
from segmentation.constant.constant import *
from segmentation.exception.exception import ApplicationException
from segmentation.entity.config_entity import *
from segmentation.entity.artifact_entity import *
from segmentation.components.data_ingestion import DataIngestion




class Pipeline():

    def __init__(self,training_pipeline_config=TrainingPipelineConfig()):
        try:
            
            self.training_pipeline_config=training_pipeline_config
            
        except Exception as e:
            raise ApplicationException(e, sys) from e

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config=DataIngestionConfig(self.training_pipeline_config))
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
        
        
    def run_pipeline(self):
            try:
                #data ingestion
                data_ingestion_artifact = self.start_data_ingestion()
                

                
            except Exception as e:
                raise ApplicationException(e, sys) from e
            
