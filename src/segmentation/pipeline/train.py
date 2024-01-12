
from segmentation.constant.constant import *
from segmentation.exception.exception import ApplicationException
from segmentation.entity.config_entity import *
from segmentation.entity.artifact_entity import *
from segmentation.components.data_ingestion import DataIngestion
from segmentation.components.data_validation import DataValidation
from segmentation.components.data_transformation import DataTransformation


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
        
    def start_data_validation(self, data_ingestion_artifact:DataIngestionArtifact)-> DataValidationArtifact:
        try:
            data_validation = DataValidation(data_validation_config=DataValidationConfig(self.training_pipeline_config),
                                             data_ingestion_artifact=data_ingestion_artifact)
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    def start_data_transformation(self,data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        try:
            data_transformation = DataTransformation(
                data_transformation_config = DataTransformationConfig(self.training_pipeline_config),
                data_validation_artifact = data_validation_artifact)

            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise ApplicationException(e,sys) from e
        
    def run_pipeline(self):
            try:
                #data ingestion
                data_ingestion_artifact = self.start_data_ingestion()
                # data Validation 
                data_validation_artifact=self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
                 # data transformation 
                data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
                

                
            except Exception as e:
                raise ApplicationException(e, sys) from e
            
