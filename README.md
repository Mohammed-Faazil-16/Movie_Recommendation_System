Movie Recommendation System
This project is a Movie Recommendation System that suggests movies based on genres, keywords, taglines, cast, and directors. The system uses TF-IDF Vectorization and Cosine Similarity to find and recommend movies similar to the user's input. It also fetches and displays movie posters using the TMDB (The Movie Database) API.

Project Overview
Data Source: The system uses a dataset of movies (movies.csv), containing relevant features such as genres, keywords, tagline, cast, and director.
Recommendation Engine: The core recommendation engine is built using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization and cosine similarity.
Movie Poster Display: The project integrates with the TMDB API to fetch and display movie posters for the recommended movies.
User Authentication: Basic user authentication is implemented, where users can sign up and log in with a username and password.

Features
Movie Recommendation: Provides movie recommendations based on user input.
Poster Fetching: Fetches and displays movie posters using the TMDB API.
User Authentication: Securely handles user sign-up and login functionality using SQLite.

# Movie Recommendation System

This is a movie recommendation system that recommends movies based on genres, keywords, taglines, cast, and directors.

## Project Structure

- `data/`: Contains the dataset (`movies.csv`).
- `database/`: Contains the database file (`user.db`).
- `src/`: Contains the source code (`movie_recommendation_system.py`).
- `notebooks/`: Contains Jupyter notebooks for running the project in a notebook environment.
- `requirements.txt`: Lists all dependencies required to run the project.
- `.gitignore`: Specifies files and directories to ignore in version control.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Mohammed-Faazil-16/movie_recommendation_system.git
   cd Movie-Recommendation-System
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the script:
   ```
   python src/movie_recommendation_system.py
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
