import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.components.data_transformation import DataTransformation
from sklearn.neighbors import NearestNeighbors

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

class RecommendPipeline:
    def __init__(self):
        pass

    def recommend_movies_knn(self, movie_title):
        try:
            num_recommendations=5
            # Paths to load the model and vectorizer
            model_path = os.path.join("artifacts", "model.pkl")
           
            # Load KNN model and vectorizer from artifacts
            knn = load_object(file_path=model_path)
            

            # Load and prepare the data (tags and vector_matrix)
            data_transformation = DataTransformation()
            tags, vector_matrix = data_transformation.initiate_data_transformation('artifacts/credits.csv', 'artifacts/movies.csv')

            # Assuming 'tags' is the DataFrame with movie_id, title, and tags
            

            # Get the index of the movie based on its title
            movie_idx = tags[tags['title'].str.lower() == movie_title.lower()].index[0]

            # Get the feature vector for the input movie (tags) from the vectorized data
            movie_vector = vector_matrix[movie_idx].toarray()

            # Find the nearest neighbors
            distances, indices = knn.kneighbors(movie_vector, n_neighbors=num_recommendations + 1)  # +1 to include the movie itself

            # Get the recommended movie indices (excluding the input movie itself)
            recommended_movie_indices = indices[0][1:]  # Skip the first index (which is the movie itself)
            recommended_movies = tags.iloc[recommended_movie_indices][['movie_id', 'title']]

            return recommended_movies

        except Exception as e:
            raise CustomException(f"Error during movie recommendation: {e}", sys)
