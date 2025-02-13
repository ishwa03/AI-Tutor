import os
import sys
from flask import Flask, request, render_template
import pandas as pd
from src.pipeline.recommend_pipeline import RecommendPipeline
from src.utils import load_object

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend_movies', methods=['GET', 'POST'])
def recommend_movies():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Get the movie title from the form
        movie_title = request.form.get('movie_title')
        
        # Initialize the recommendation pipeline
        recommend_pipeline = RecommendPipeline()
        
        # Call the recommendation function
        recommended_movies = recommend_pipeline.recommend_movies_knn(movie_title = movie_title)
        
        # Convert the recommended movies to a list or JSON for rendering
        movie_titles = recommended_movies['title'].tolist()
        
        # Render the results on the web page
        return render_template('home.html', results=movie_titles)
    
    


if __name__ == "__main__":
    app.run(host="0.0.0.0")
