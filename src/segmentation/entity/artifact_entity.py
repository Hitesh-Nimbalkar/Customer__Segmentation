from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_file_path:str 
    
    
@dataclass
class DataValidationArtifact:
    validated_train_path:str

@dataclass
class DataTransformationArtifact:
    feature_eng_df:str
    transformed_train_file_path:str
    feature_engineering_object_file_path: str
    
@dataclass
class ModelTrainerArtifact:
    model_selected:str
    report_path:str
    model_prediction_png:str
    