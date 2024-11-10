
import numpy as np  
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import sqlite3
import hashlib

# Database setup
conn = sqlite3.connect('movie_recommendations.db')
c = conn.cursor()

# Create necessary tables
c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)''')
c.execute('''CREATE TABLE IF NOT EXISTS ratings (username TEXT, movie_title TEXT, rating INTEGER)''')
c.execute('''CREATE TABLE IF NOT EXISTS feedback (username TEXT, movie_title TEXT, feedback TEXT)''')
conn.commit()

# Hashing function for passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Sign-up function
def sign_up():
    username = input("Enter a username: ")
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    if c.fetchone():
        print("Username already exists! Please try again.")
        return sign_up()

    password = input("Enter a password: ")
    hashed_password = hash_password(password)
    c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
    conn.commit()
    print("User registered successfully!")

# Sign-in function
def sign_in():
    username = input("Enter your username: ")
    password = input("Enter your password: ")
    hashed_password = hash_password(password)
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, hashed_password))
    if c.fetchone():
        print("Login successful!")
        return username  # Return username to identify the logged-in user
    else:
        print("Invalid credentials! Please try again.")
        return sign_in()

# Ask the user to sign in or sign up
def authenticate_user():
    print("Welcome! Please sign in or sign up to use the Movie Recommendation System.")
    while True:
        action = input("Type 'sign in' to log in or 'sign up' to register: ").lower()
        if action == 'sign in':
            username = sign_in()
            break
        elif action == 'sign up':
            sign_up()
        else:
            print("Invalid option, please type 'sign in' or 'sign up'.")
    return username

# Data Collection and Pre-Processing
def load_and_process_data(file_path):
    # Loading the data from the CSV file to a pandas dataframe
    movies_data = pd.read_csv(file_path, low_memory=False)

    # Ensure titles are strings to avoid TypeError in difflib.get_close_matches
    movies_data['title'] = movies_data['title'].astype(str)

    # Selecting the relevant features for recommendation
    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

    # Replacing null values with an empty string
    for feature in selected_features:
        movies_data[feature] = movies_data[feature].fillna('')

    # Combining all selected features into one string
    combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

    # Converting the text data to feature vectors
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)

    return movies_data, feature_vectors

# Getting movie recommendations based on cosine similarity
def get_movie_recommendations(movie_name, movies_data, feature_vectors, username):
    # Finding the movie in the dataset
    movie_list = movies_data['title'].tolist()
    close_matches = difflib.get_close_matches(movie_name, movie_list, n=5, cutoff=0.5)
    
    if not close_matches:
        print("No matching movies found.")
        return []

    movie_index = movies_data[movies_data['title'] == close_matches[0]].index[0]
    
    # Calculate the cosine similarity
    similarity_scores = cosine_similarity(feature_vectors[movie_index], feature_vectors).flatten()

    # Get top 5 movie recommendations based on cosine similarity
    similar_movies = sorted(list(enumerate(similarity_scores)), key=lambda x: x[1], reverse=True)[1:6]
    
    recommended_movies = []
    for movie in similar_movies:
        recommended_movies.append(movies_data['title'].iloc[movie[0]])

    return recommended_movies, similarity_scores[movie_index].flatten()

# TMDB API setup
API_KEY = '31b1e60b9e5ec9908ece9580a1aea0ab'
BASE_URL = 'https://api.themoviedb.org/3'

# Function to get the movie poster URL
def get_movie_poster(movie_name):
    search_url = f"{BASE_URL}/search/movie"
    params = {'api_key': API_KEY, 'query': movie_name}
    response = requests.get(search_url, params=params)
    data = response.json()

    if data['results']:
        movie = data['results'][0]
        poster_path = movie.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w300{poster_path}"
    return None

# Function to display movie posters and titles with enhanced layout
def display_movie_posters(movie_list):
    n = len(movie_list)
    fig, axes = plt.subplots(n, 1, figsize=(8, n * 3))  # Improved layout

    for index, movie_name in enumerate(movie_list):
        poster_url = get_movie_poster(movie_name)
        if poster_url:
            response = requests.get(poster_url)
            img = Image.open(BytesIO(response.content))

            axes[index].imshow(img)
            axes[index].axis('off')
            axes[index].set_title(movie_name, fontsize=12, color='darkblue')  # Enhanced title styling

    plt.tight_layout()
    plt.show()

# Visualize the similarity scores of recommended movies
def plot_similarity_scores(similarity_scores, movie_name):
    plt.figure(figsize=(10, 6))
    plt.barh(range(1, len(similarity_scores) + 1), similarity_scores, color='skyblue')
    plt.yticks(range(1, len(similarity_scores) + 1), [f"Movie {i+1}" for i in range(len(similarity_scores))])
    plt.title(f"Cosine Similarity Scores for '{movie_name}'", fontsize=14)
    plt.xlabel('Cosine Similarity', fontsize=12)
    plt.ylabel('Recommended Movies', fontsize=12)
    plt.show()

# Plot innovative graphs: Movie Ratings vs Runtime & Genre-wise Popularity
def plot_innovative_graphs(movies_data, recommended_movies):
    # Filter data for the recommended movies
    recommended_data = movies_data[movies_data['title'].isin(recommended_movies)]

    # Scatter plot: Movie Ratings vs Runtime
    plt.figure(figsize=(10, 6))
    plt.scatter(recommended_data['runtime'], recommended_data['vote_average'], color='orange')
    plt.title("Movie Ratings vs Runtime", fontsize=14)
    plt.xlabel("Runtime (Minutes)", fontsize=12)
    plt.ylabel("Average Rating", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Bar plot: Genre-wise Popularity (Count of movies by genre)
    genre_counts = recommended_data['genres'].str.split('|').explode().value_counts()
    plt.figure(figsize=(10, 6))
    genre_counts.plot(kind='bar', color='lightcoral')
    plt.title("Genre-wise Popularity", fontsize=14)
    plt.xlabel("Genres", fontsize=12)
    plt.ylabel("Number of Movies", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Collecting user feedback
def collect_feedback(username, recommended_movies):
    for movie in recommended_movies:
        feedback = input(f"Did you like the movie {movie}? (yes/no): ").lower()
        c.execute("INSERT INTO feedback (username, movie_title, feedback) VALUES (?, ?, ?)", (username, movie, feedback))
        conn.commit()

# Displaying feedback statistics
def display_feedback_statistics(username):
    c.execute("SELECT movie_title, feedback FROM feedback WHERE username = ?", (username,))
    feedback_data = c.fetchall()

    feedback_dict = {"liked": 0, "disliked": 0}
    
    for feedback in feedback_data:
        if feedback[1] == "yes":
            feedback_dict["liked"] += 1
        elif feedback[1] == "no":
            feedback_dict["disliked"] += 1

    print(f"Feedback Summary for {username}:")
    print(f"Movies Liked: {feedback_dict['liked']}")
    print(f"Movies Disliked: {feedback_dict['disliked']}")

# Main function to execute the recommendation system
def run_movie_recommendation_system():
    movie_data_file = 'movies.csv'  # Replace with actual path to your 'movies.csv'
    movies_data, feature_vectors = load_and_process_data(movie_data_file)

    username = authenticate_user()

    while True:
        movie_name = input("Enter a movie name to get recommendations (or 'exit' to quit): ")
        if movie_name.lower() == 'exit':
            break

        recommended_movies, similarity_scores = get_movie_recommendations(movie_name, movies_data, feature_vectors, username)

        if recommended_movies:
            display_movie_posters(recommended_movies)
            plot_similarity_scores(similarity_scores, movie_name)
            plot_innovative_graphs(movies_data, recommended_movies)
            collect_feedback(username, recommended_movies)
            display_feedback_statistics(username)

run_movie_recommendation_system()
