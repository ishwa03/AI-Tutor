import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from sklearn.neighbors import NearestNeighbors
 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class Modeltrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig() 
    
    def initiate_model_trainer(self, vector_matrix):
        try:
            logging.info("Training Initiated")
            knn = NearestNeighbors(n_neighbors=6, metric='cosine')  # n_neighbors=6 to include the input movie itself
            knn.fit(vector_matrix)  # Fit on the movie tag vectors

            logging.info(f"KNN model trained successfully")

            # Save the trained KNN model
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=knn)

            # Return the trained KNN model and vectorizer for later use
            return knn

        except Exception as e:
            raise CustomException(f"Error during model training: {e}", sys)