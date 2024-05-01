# Import modules
import numpy as np
import pandas as pd
import nltk

# Set seed for reproducibility
np.random.seed(5)

# Read in IMDb and Wikipedia movie data (both in same file)
movies_df = pd.read_csv('datasets/netflix_shows.csv')

nltk.download('punkt')
# Tokenize a paragraph into sentences and store in sent_tokenized
sent_tokenized = [sent for sent in nltk.sent_tokenize("""
                        Today (May 19, 2016) is his only daughter's wedding. 
                        Vito Corleone is the Godfather.
                        """)]

# Word Tokenize first sentence from sent_tokenized, save as words_tokenized
words_tokenized = [word for word in nltk.word_tokenize(sent_tokenized[0])]

# Remove tokens that do not contain any letters from words_tokenized
import re

filtered = [word for word in words_tokenized if re.search('[a-zA-Z]', word)]

from nltk.stem import SnowballStemmer
# Initialize the Snowball stemmer for English
snowball_stemmer = SnowballStemmer('english')

stemmed_words = [snowball_stemmer.stem(word) for word in filtered]

def tokenize_and_stem(text):
    sent_tokenized = [sent for sent in nltk.sent_tokenize(text)]
    words_tokenized = [word for word in nltk.word_tokenize(sent_tokenized[0])]
    filtered = [word for word in words_tokenized if re.search('[a-zA-Z]', word)]
    stemmed_words = [snowball_stemmer.stem(word) for word in filtered]
    return stemmed_words


from tfIdfInheritVectorizer.feature_extraction.vectorizer import TFIDFVectorizer as TfidfVectorizer
# Might want to switch to sklearn here? Has way more documentation
tfidf_vectorizer = TfidfVectorizer(max_df=1.0, max_features=200000,
                                 min_df=0.0, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem,
                                 ngram_range=(1,3))
tfidf_matrix = tfidf_vectorizer.fit_transform([x for x in movies_df["description"]])

# Import k-means to perform clusters
from sklearn.cluster import KMeans
km = KMeans(n_clusters=4)

km.fit(tfidf_matrix)
clusters = km.labels_.tolist()
movies_df["cluster"] = clusters
cluster_counts = movies_df['cluster'].value_counts()

from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns


# Calculate the similarity distance
similarity_distance = 1 - cosine_similarity(tfidf_matrix)

import numpy as np

cosine_sim_matrix = cosine_similarity(tfidf_matrix)
# Exclude self-similarity comparisons (do not compare a document to itself)
np.fill_diagonal(cosine_sim_matrix, 0)
# Get indices for top right of matrix (except main diagonal)
upper_triangular_indices = np.triu_indices(cosine_sim_matrix.shape[0], k=1)
# Get according similarities values generated previously
similarities_flat = cosine_sim_matrix[upper_triangular_indices]
#Sort to get top 20 most similar shows
top_20_indices = similarities_flat.argsort()[-20:][::-1]

most_similar_indices = (upper_triangular_indices[0][top_20_indices], upper_triangular_indices[1][top_20_indices])

for i, (idx1, idx2) in enumerate(zip(*most_similar_indices), 1):
    movie1_title = movies_df.iloc[idx1]["title"]
    movie2_title = movies_df.iloc[idx2]["title"]
    print(f"{i}. {movie1_title} and {movie2_title}")

def find_similar_movies(input_description):
    # Preprocess input description
    input_description_stemmed = tokenize_and_stem(input_description)
    input_description_transformed = ' '.join(input_description_stemmed)
    
    # Transform input description into TF-IDF vector
    input_tfidf_vector = tfidf_vectorizer.transform([input_description_transformed])
    
    # Calculate cosine similarity between input description and dataset descriptions
    similarity_scores = cosine_similarity(input_tfidf_vector, tfidf_matrix)
    
    # Get indices of top similar movies
    top_indices = similarity_scores.argsort()[0][-5:][::-1]
    
    # Get titles and descriptions of top similar movies
    similar_movies_data = []
    for idx in top_indices:
        title = movies_df.iloc[idx]['title']
        description = movies_df.iloc[idx]['description']
        similar_movies_data.append((title, description))
    
    
    return similar_movies_data


