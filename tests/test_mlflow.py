
import os
from lightning.pytorch.loggers import MLFlowLogger

def test_mlflow_logger():
    tracking_uri = "file:./results/mlruns"
    try:
        logger = MLFlowLogger(
            experiment_name="Test",
            run_name="TestRun",
            tracking_uri=tracking_uri
        )
        print(f"Logger tracking_uri: {logger.tracking_uri}")
        print(f"Type of tracking_uri: {type(logger.tracking_uri)}")
        
        # Test accessing it
        uri = logger.tracking_uri
        if callable(uri):
            print("Warning: tracking_uri is CALLABLE!")
        
    except Exception as e:
        print(f"Failed to init MLFlowLogger: {e}")

if __name__ == "__main__":
    test_mlflow_logger()
