
from segmentation.entity.config_entity import *
from segmentation.entity.artifact_entity import *
import shutil


class Model_pusher:
    def __init__(self,model_evaluation_artifact:ModelEvaluationArtifact) -> None:
        
        self.model_evaluation_artifact=model_evaluation_artifact
        
        self.model_path=self.model_evaluation_artifact.model_file_path
        self.model_report=self.model_evaluation_artifact.model_report
        self.model_eval_png=self.model_evaluation_artifact.prediction_png
        
        
        saved_model_config=SavedModelConfig()
        self.saved_model_object=saved_model_config.saved_model_object_path
        self.saved_model_report=saved_model_config.saved_model_report_path
        self.saved_png=saved_model_config.saved_model_png
        
        
    def start_model_pusher(self):
        
           
        # Copying Model 
        shutil.copy(self.model_path, self.saved_model_object)
        # Copying Report 
        shutil.copy(self.model_report, self.saved_model_report)
        
        shutil.copy(self.model_report,'params.yaml')
        
        shutil.copy(self.model_eval_png, self.saved_png)
        
        
        model_pusher_artifact= ModelPusherArtifact(message="Training Pipeline Complete")
       
        return model_pusher_artifact
        
        
        



        