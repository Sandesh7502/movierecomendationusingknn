import requests
from flask import Flask, render_template, request, session, redirect, url_for, flash
import pickle
import numpy as np
import pandas as pd
import sqlite3

app = Flask(__name__)


def check_user(username, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    print(user)
    conn.close()
    if user and user[2] == password:
        return True
    else:
        return False


app.secret_key = 'secret_key'


# Load data

@app.route('/')
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    username = session.get('username')
    return render_template('index.html', username=username)


@app.route('/admin.html')
def admin_page():
    username = session.get('username')
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    if session.get('username') != 'admin':
        flash('Only admin users can access the admin page', 'error')
        return redirect(url_for('index'))

        # Create connection to database
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    # Fetch all users from database
    cursor.execute('SELECT * FROM users')
    users = cursor.fetchall()

    # Close connection to database
    conn.close()

    return render_template('admin.html', users=users, username=username)


@app.route('/delete_user', methods=['POST'])
def delete_user():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    if session.get('username') != 'admin':
        flash('Only admin users can delete users', 'error')
        return redirect(url_for('index'))

    # Get form data
    username = request.form['username']

    # Create connection to database
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    # Delete user from database
    cursor.execute('DELETE FROM users WHERE username = ?', (username,))
    conn.commit()

    # Close connection to database
    conn.close()

    flash(f'{username} has been deleted', 'message')
    return redirect(url_for('admin_page'))


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Get form data
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # Create connection to database
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()

        # Insert user data into database
        cursor.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)', (username, email, password))
        conn.commit()

        # Close connection to database
        conn.close()

        print('Sign up successful, please log in.')
        return redirect(url_for('login'))

    return render_template('signup.html', login_url=url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Check if username and password are valid
        username = request.form['username']
        password = request.form['password']
        if check_user(username, password):
            # Set session variable to indicate that user is logged in
            session['logged_in'] = True
            session['username'] = request.form['username']
            return redirect(url_for('index'))
        else:
            # Invalid login, show error message
            flash('Please enter valid username or password', 'error')

    # Render login page template
    return render_template('login.html')


@app.route('/recommend.html')
def recommend_ui():
    username = session.get('username')
    return render_template('recommend.html', username=username)


# Load the movies dataset
with open('movies.pkl', 'rb') as file:
    movies = pickle.load(file)
with open('reducedfeature_vectors.pkl', 'rb') as file:
    reducedfeature_vectors = pickle.load(file)


# Define a function to calculate cosine similarity between two vectors
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Define the Knn_model function to find the nearest neighbors of a given movie
def Knn_model(movie_id, reducedfeature_vectors, k):
    # Get the feature vector for the given movie
    movie_vector = reducedfeature_vectors[movie_id]

    # Compute the cosine similarity between the given movie and all other movies
    similarities = []
    for i, vector in enumerate(reducedfeature_vectors):
        if i != movie_id:
            sim = cosine_similarity(movie_vector, vector)
            similarities.append((i, sim))

    # Sort the movies by similarity and return the top k
    similarities.sort(key=lambda x: x[1], reverse=True)
    neighbors = similarities[:k]
    return neighbors


# Define the recommend_movies function to generate recommendations for a given movie title
def recommend_movies(title, reducedfeature_vectors, movies):
    # Check if the movie exists in the movies dataframe
    if title not in movies['title'].unique():
        return {'error': 'Sorry, we could not find the movie you are looking for.'}

    # Find the movie ID of the given title
    movie_id = movies[movies['title'] == title].index[0]

    # Find the 5 nearest neighbors of the given movie
    neighbors = Knn_model(movie_id, reducedfeature_vectors, 5)

    # Prepare the list of recommended movies
    recommended_movies = []
    for neighbor in neighbors:
        recommended_movies.append(movies.loc[neighbor[0], 'title'])

    return recommended_movies


# Define the Flask route to handle movie recommendations
@app.route('/recommend_movies', methods=['POST'])
def get_recommendations():
    username = session.get('username')
    # Get the movie title from the form data
    title = request.form['movie_title']

    # Call the recommend_movies function to get recommendations
    recommendations = recommend_movies(title, reducedfeature_vectors, movies)

    return render_template('recommend.html', username=username, recommendations=recommendations[0])


if __name__ == '__main__':
    app.run(debug=True)
