#Import necessary libraries
import numpy as np
import pandas as pd


#Load data and display initial rows
credits = pd.read_csv("dataset/credits.csv")
movies = pd.read_csv("dataset/movies.csv")
credits.head()
movies.head()

#Merge credits dataframe with movies dataframe on 'title' column
movies = movies.merge(credits, on='title')
movies.info()

#Select only relevant columns from merged dataframe
movies = movies[['id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.info()



#Remove rows with missing data
#Remove any duplicated rows based on 'title' column and keep first occurrence of each title
movies.dropna(inplace=True)
movies.duplicated().sum()
movies.isnull().sum()

#Convert 'genres', 'cast', 'crew', and 'keywords' columns to lists and remove spaces in the list elements
import ast

# Define a function to convert stringified lists to actual lists of values
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L


movies['genres'] = movies['genres'].apply(convert)
movies.head()
movies['keywords'] = movies['keywords'].apply(convert)
movies.head()


def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter += 1
    return L


movies['cast'] = movies['cast'].apply(convert)
movies.head()
movies['cast'] = movies['cast'].apply(lambda x: x[0:3])


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L


movies['crew'] = movies['crew'].apply(fetch_director)
movies.sample()
movies.head()


def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ", ""))
    return L1


movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)
movies.head()
# Remove rows with missing data
movies.dropna(inplace=True)
duplicates = movies[movies.duplicated(['title'], keep=False)]
if not duplicates.empty:
    print(f"There are {len(duplicates)} movies with the same name:")
    print(duplicates['title'])
    movies = movies.drop_duplicates(subset='title', keep=False)
else:
    print("There are no movies with the same name.")

# Create a MultiLabelBinarizer object to transform the categorical data into binary form
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()

# Fit the object to the categorical data
genres = mlb.fit_transform(movies["genres"])
cast = mlb.fit_transform(movies["cast"])
crew = mlb.fit_transform(movies["crew"])

# Create a TF-IDF vectorizer object
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words="english")

# Join the list of keywords into a single string for each movie
keywords_strings = [' '.join(keyword_list) for keyword_list in movies['keywords']]

# Transform the keywords column into TF-IDF vectors
keywords_tfidf = tfidf.fit_transform(keywords_strings).todense()

# Transform the 'overview' column into TF-IDF vectors
overviews_tfidf = tfidf.fit_transform(movies["overview"]).todense()

# Combine binary categorical features with TF-IDF vectors
categorical_features = pd.DataFrame(np.concatenate([genres, cast, crew], axis=1))
textual_features = pd.DataFrame(np.concatenate([overviews_tfidf, keywords_tfidf], axis=1))
feature_vectors = pd.concat([categorical_features, textual_features], axis=1)
feature_vectors.insert(0, 'movie_id', movies['id'])

# Set the movie_id column as the index
feature_vectors = feature_vectors.set_index("movie_id")
print(feature_vectors.columns)
print(feature_vectors.shape)

#Perform PCA on the feature vectors and plot eigenvalues to select number of components
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# create a PCA object
pca = PCA()

# fit the PCA model to the feature vectors
pca.fit(feature_vectors)

# get the eigenvalues
eigenvalues = pca.explained_variance_

# get the proportion of variance explained by each component
explained_variance_ratio = pca.explained_variance_ratio_

# compute the cumulative sum of the explained variance
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# plot the eigenvalues
plt.plot(eigenvalues)

# plot the cumulative sum of the eigenvalues
plt.plot(cumulative_variance_ratio)

# compute the mean eigenvalue
mean_eigenvalue = np.mean(eigenvalues)

# compute the expected eigenvalues under the broken-stick criterion
expected_eigenvalues = np.zeros_like(eigenvalues)
for i in range(len(eigenvalues)):
    expected_eigenvalues[i] = np.sum([(j / i) ** i for j in range(i)])

# plot the mean eigenvalue and the expected eigenvalues
plt.axhline(mean_eigenvalue, color='r', linestyle='--')
plt.plot(expected_eigenvalues)

# use the Kaiser criterion to select the number of components
kaiser_components = np.where(eigenvalues > mean_eigenvalue)[0]
print('Number of components selected by Kaiser criterion:', len(kaiser_components))

pca = PCA(n_components=1013)

pca.fit(feature_vectors)

reducedfeature_vectors = pca.transform(feature_vectors)

#Building KNN Model
# Define a function to calculate cosine similarity between two vectors
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
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


# Find the 5 nearest neighbors of the movie with ID 0
neighbors = Knn_model(0, reducedfeature_vectors, 5)
for neighbor in neighbors:
    print(movies.iloc[neighbor[0]]['title'])


def recommend_movies(title, reducedfeature_vectors, movies):
    # Check if the movie exists in the movies dataframe
    if title not in movies['title'].unique():
        print('Sorry, we could not find the movie you are looking for.')
        return

    # Find the movie ID of the given title
    movie_id = movies[movies['title'] == title].index[0]

    # Find the 5 nearest neighbors of the given movie
    neighbors = Knn_model(movie_id, reducedfeature_vectors, 5)

    # Prepare the list of recommended movies
    recommended_movies = []
    for neighbor in neighbors:
        movie_info = movies.iloc[neighbor[0]]
        recommended_movies.append({
            'Title': movie_info['title'],
            'cosinesimilarity_score': neighbor[1],
            'Genres': movie_info['genres'],
            'Cast': movie_info['cast']
        })

    # Sort the recommended movies by distance
    recommended_movies = sorted(recommended_movies, key=lambda x: x['cosinesimilarity_score'], reverse=True)

    # Convert the recommended_movies list to a pandas DataFrame
    recommended_movies_df = pd.DataFrame(recommended_movies)

    return recommended_movies_df


recommend_movies('Avatar', reducedfeature_vectors, movies)


import pickle

# Save the trained model as a pickle file
with open('Knn_model.pkl', 'wb') as file:
    pickle.dump(Knn_model, file)

# Save the trained model as a pickle file
with open('reducedfeature_vectors.pkl', 'wb') as file:
    pickle.dump(reducedfeature_vectors, file)

# Save the trained model as a pickle file
with open('movies.pkl', 'wb') as file:
    pickle.dump(movies, file)