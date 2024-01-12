from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_file_path:str 
    
    
@dataclass
class DataValidationArtifact:
    validated_train_path:str

@dataclass
class DataTransformationArtifact:
    transformed_train_file_path:str
    feature_engineering_object_file_path: str
    