
from segmentation.exception.exception import ApplicationException
from segmentation.logger.logger import logging
import sys
from segmentation.utils.main_utils import *
from segmentation.entity.config_entity import *
from segmentation.entity.artifact_entity import *
from segmentation.constant import *
from segmentation.utils.main_utils import read_yaml_file,load_object
from segmentation.constant import *
import shutil





class ModelEvaluation:


    def __init__(self,model_evaluation_config:ModelEvalConfig,
                    model_trainer_artifact:ModelTrainerArtifact):
        
        try:
            self.model_trainer_artifact=model_trainer_artifact
            
            # Model Evaluation Config
            self.model_evaluation_config=model_evaluation_config
            self.model_evaluation_directory=self.model_evaluation_config.model_eval_directory
            os.makedirs(self.model_evaluation_directory,exist_ok=True)
            
            # Saved Model Config
            self.saved_model_config=SavedModelConfig()
            self.saved_model_directory=self.saved_model_config.saved_model_dir

            
        except Exception as e:
            raise ApplicationException(e,sys)
        
        
    def save_model_and_params(self,saved_model_path, report_file_path, best_params,best_model):
        # Save the best parameters as a YAML file
        with open(report_file_path, 'w') as yaml_file:
            yaml.dump(best_params, yaml_file, default_flow_style=False)
        logging.info(f"Best model parameters saved to: {report_file_path}")

        # Save the best model
        save_object(file_path=saved_model_path,obj=best_model)
        logging.info(f"Best model saved to: {saved_model_path}")


    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:

            logging.info(" Model Evaluation Started ")
            
            
            # Saved Model files
            artifact_model_path = self.model_trainer_artifact.model_selected
            artifact_report_path=self.model_trainer_artifact.report_path
            artifact_png=self.model_trainer_artifact.model_prediction_png
            
            logging.info(f"{artifact_png}")

            
            model_eval_model_path=self.model_evaluation_config.model_eval_object
            model_eval_model_report=self.model_evaluation_config.model_eval_report
            model_eval_png=self.model_evaluation_config.model_eval_png
            
            logging.info(f"{model_eval_png}")
            
            saved_model_directory=self.saved_model_directory
            os.makedirs(saved_model_directory,exist_ok=True)
            
            if not os.listdir(saved_model_directory):
                logging.info("No data found in saved model Directory")
                logging.info('Copying content from Artifacto to Model Eval Artifact....')
                # Copying Model and Report to saved model directory 
                # Copying Model 
                shutil.copy(artifact_model_path, model_eval_model_path)
                # Copying Report 
                shutil.copy(artifact_report_path, model_eval_model_report)
                
                shutil.copy(artifact_png, model_eval_png)
                
            else: 
                saved_directory_model_report=self.saved_model_config.saved_model_report_path
                saved_model_path=self.saved_model_config.saved_model_object_path
                saved_model_png=self.saved_model_config.saved_model_png
                
                logging.info(f" saved Report path{saved_directory_model_report} ")
                saved_model_report=read_yaml_file(saved_directory_model_report)
                saved_model_score=float(saved_model_report['R2_score'])
                
                artifact_report_data=read_yaml_file(artifact_report_path)
                artifact_model_score=float(artifact_report_data['R2_score'])
            
                
                if saved_model_score > artifact_model_score:
                    # If the saved model score is higher than the artifact model score
                    
                    shutil.copy(saved_model_path, model_eval_model_path)
                     # Copying Report 
                    shutil.copy(saved_model_path, model_eval_model_report)
                    
                    shutil.copy(saved_model_png, model_eval_png)
                    
                
                    logging.info(f"Selected saved model for training as its score ({saved_model_score}) is higher than the artifact model score ({artifact_model_score})")
                    print(f"Selected saved model for training as its score ({saved_model_score}) is higher than the artifact model score ({artifact_model_score})")
                elif saved_model_score < artifact_model_score:
                    # If the saved model score is lower than the artifact model score
                    shutil.copy(artifact_model_path, model_eval_model_path)
                    # Copying Report
                    shutil.copy(artifact_report_path, model_eval_model_report)
                    
                    shutil.copy(artifact_png, model_eval_png)
                    

                    logging.info(f"Selected artifact model for training as its score ({artifact_model_score}) is higher than the saved model score ({saved_model_score})")
                    print(f"Selected artifact model for training as its score ({artifact_model_score}) is higher than the saved model score ({saved_model_score})")
                else: 
                                        
                    shutil.copy(artifact_model_path, model_eval_model_path)
                    # Copying Report
                    shutil.copy(artifact_report_path, model_eval_model_report)
                    
                    shutil.copy(artifact_png, model_eval_png)
                    
                    logging.info(" Both Models have similar score Storing model with new artifacteters  ")
            
            
            model_evaluation_artifact=ModelEvaluationArtifact(message="Model Evaluation complete",
                                                            model_report=model_eval_model_path,
                                                            model_file_path=model_eval_model_report,
                                                            prediction_png=model_eval_png)

            return model_evaluation_artifact
        

 


    def __del__(self):
        logging.info(f"\n{'*'*20} Model evaluation log completed {'*'*20}\n\n")
        
        