from src.segmentation.logger.logger import logging
from src.segmentation.exception.exception import ApplicationException
from src.segmentation.pipeline.train import Pipeline
import sys,os

def main():
    try:
        pipeline = Pipeline()
        pipeline.run_pipeline()


    except Exception as e:
        raise ApplicationException(e, sys) from e 

if __name__ == "__main__":
    main()