import os
import sys
import pandas as pd
from dataclasses import dataclass
import logging

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConifg

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import Modeltrainer

# Custom Exception
class CustomException(Exception):
    def __init__(self, error_message: str):
        super().__init__(error_message)

# Logging Configuration
logging.basicConfig(level=logging.INFO)

# Configuration class to specify paths for data ingestion
@dataclass
class DataIngestionConfig:
    raw_data_path_credits: str = os.path.join('artifacts', "credits.csv")
    raw_data_path_movies: str = os.path.join('artifacts', "movies.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion method.")
        try:
            # Define paths to the source CSV files
            credits_data_path = 'notebook/data/credits.csv'
            movies_data_path = 'notebook/data/movies.csv'

            # Check if files exist
            if not os.path.exists(credits_data_path):
                raise CustomException(f"{credits_data_path} file not found.")
            if not os.path.exists(movies_data_path):
                raise CustomException(f"{movies_data_path} file not found.")

            # Read the data from CSV files
            df_credits = pd.read_csv(credits_data_path)
            df_movies = pd.read_csv(movies_data_path)
            logging.info('Data loaded successfully into DataFrames.')

            # Create directories if they do not exist
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path_credits), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path_movies), exist_ok=True)

            # Save the data into 'artifacts' directory
            df_credits.to_csv(self.ingestion_config.raw_data_path_credits, index=False)
            df_movies.to_csv(self.ingestion_config.raw_data_path_movies, index=False)
            logging.info("Ingestion completed and data saved to artifacts.")

            # Return the file paths of the ingested data
            return self.ingestion_config.raw_data_path_credits, self.ingestion_config.raw_data_path_movies

        except Exception as e:
            logging.error(f"Error during data ingestion: {e}")
            raise CustomException(str(e))


# Run the ingestion process
if __name__ == "__main__":
    try:
        data_ingestion_obj = DataIngestion()
        credits_data, movies_data = data_ingestion_obj.initiate_data_ingestion()

        data_transformation = DataTransformation()
        movie_tags, tfidf_matrix = data_transformation.initiate_data_transformation(credits_data, movies_data)

        model_trainer = Modeltrainer()
        model = model_trainer.initiate_model_trainer(tfidf_matrix)

        # Print the paths of ingested data
        print(f"Data Ingested Successfully: \nCredits Data Path: {credits_data}\nMovies Data Path: {movies_data}")

    except CustomException as ce:
        logging.error(f"CustomException: {ce}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
