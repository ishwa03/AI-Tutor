import os
import sys
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

@dataclass
class DataTransformationConifg:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConifg()

    def prepare_movies_data(self, movies, credits):
        """
        Merges movies and credits data, preprocesses, and prepares tags for TF-IDF vectorization
        """
        try:
            # Merging movies with credits on 'title'
            merged = movies.merge(credits, on='title')

            # Selecting relevant columns
            merged = merged[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

            # Function to convert string representation of lists to actual lists
            def convert(text):
                L = []
                for i in ast.literal_eval(text):
                    L.append(i['name']) 
                return L 

            # Drop missing values
            merged.dropna(inplace=True)

            # Apply conversion to genres and keywords columns
            merged['genres'] = merged['genres'].apply(convert)
            merged['keywords'] = merged['keywords'].apply(convert)

            # Function to convert cast
            def convert3(text):
                L = []
                counter = 0
                for i in ast.literal_eval(text):
                    if counter < 3:  # Only take top 3 cast members
                        L.append(i['name'])
                    counter += 1
                return L
            merged['cast'] = merged['cast'].apply(convert3)

            # Reduce cast to top 3 members
            merged['cast'] = merged['cast'].apply(lambda x: x[0:3])

            # Function to fetch director from the crew list
            def fetch_director(text):
                L = []
                for i in ast.literal_eval(text):
                    if i['job'] == 'Director':
                        L.append(i['name'])
                return L

            merged['crew'] = merged['crew'].apply(fetch_director)

            # Clean-up: Remove spaces in cast, crew, genres, keywords
            def collapse(L):
                L1 = []
                for i in L:
                    L1.append(i.replace(" ", ""))
                return L1

            merged['cast'] = merged['cast'].apply(collapse)
            merged['crew'] = merged['crew'].apply(collapse)
            merged['genres'] = merged['genres'].apply(collapse)
            merged['keywords'] = merged['keywords'].apply(collapse)

            # Split the 'overview' into words and combine it with other columns to create a "tags" column
            merged['overview'] = merged['overview'].apply(lambda x: x.split())
            merged['tags'] = merged['overview'] + merged['genres'] + merged['keywords'] + merged['cast'] + merged['crew']

            # Drop original columns and only keep the 'tags' column along with 'movie_id' and 'title'
            merged = merged.drop(columns=['overview', 'genres', 'keywords', 'cast', 'crew'])

            # Convert list of tags into a string format for vectorization
            merged['tags'] = merged['tags'].apply(lambda x: " ".join(x))

            logging.info("Data preparation completed successfully.")
            return merged

        except Exception as e:
            raise CustomException(f"Error while preparing movies data: {e}", sys)

    def initiate_data_transformation(self, credit_path, movies_path):
        try:
            logging.info("Initiating data transformation")
            
            # Load the movie and credits data from the given paths
            movies = pd.read_csv(movies_path)
            credits = pd.read_csv(credit_path)

            logging.info("Read movie and credits data completed")

            # Prepare and preprocess the movies data
            transformed_movies = self.prepare_movies_data(movies, credits)

            # TF-IDF Vectorization of the 'tags' column
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(transformed_movies['tags'])

            logging.info("TF-IDF transformation completed")

            # Save the preprocessed vectorizer object
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=vectorizer)

            return transformed_movies, tfidf_matrix

        except Exception as e:
            raise CustomException(f"Error during data transformation: {e}", sys)

