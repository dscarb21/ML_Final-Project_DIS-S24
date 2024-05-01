# Analyzing Netflix TV Shows Similarity via Plot Summaries
- **Group Members:** Pedro Cruz, Sydney Eriksson, David Scarborough
- **Professor:** Lucy McCarren
- **Class:** Machine Learning B - DIS Spring 2024

## Project Overview
The task of predicting similarity among TV shows holds significant importance as it can enhance the performance of algorithms in delivering personalized media to users. Our project aims to predict the similarity between Netflix shows based on their plots, contributing to the advancement of machine learning in media recommendation systems.

### Project Goal
Can we find similar Netflix shows by analyzing similarity of show descriptions? 
### Solution Idea
Fit different Netflix shows in a number of different clusters based on the most important words of their plot descriptions.

## Dataset

> [!NOTE]
> This dataset is from 2021. Using a more up-to-date dataset would be ideal for a more powerful model.

[Netflix TV Shows 2021](https://www.kaggle.com/datasets/muhammadkashif724/netflix-tv-shows-2021) (Source: Kaggle/Muhammad Kashif). Netflix is an American subscription streaming service. The service primarily distributes original and acquired films and television shows from various genres, and it is available internationally in multiple languages. 
- Number of titles: 2662 unique values.
- License: MIT.
- Collaborators: Muhammad Kashif (Owner).

## What is Natural Language Processing?

> [!IMPORTANT]
> This project uses natural language as its primary input data type. Thus, we need to use Natural Language Processing techniques to handle data in this project.

Natural language processing, or NLP, combines computational linguistics—rule-based modeling of human language—with statistical and machine learning models to enable computers and digital devices to recognize, understand and generate text and speech (IBM).
Several NLP tasks break down human text and voice data in ways that help the computer make sense of what it's ingesting. Some of these tasks include speech recognition, part of speech tagging, word sense disambiguation, translation, etc.

## Project Implementation Steps

- Data preprocessing: Turn natural language data into numerical values that can be understood by a computational model. Methods used:
 -  Tokenization & Stemming
 - Filter out unnecessary tokens (punctuation, stopwords, LDA visualization)
 - Filter out widespread words among documents
 - Vectorize words and sentences
- Clustering: Use K-Means Clustering to group shows with most similar plots

## App Demo: Finding Similar Shows 

We've encapsulated our model within a user-friendly front-end application, which allows users to receive recommendations for 5 similar titles based on a user-provided show description. This application leverages the identical model trained on the same dataset outlined in our Jupyter notebook, ensuring consistent results. Please note that recommendations are limited to the 2021 Netflix catalogue.

### How to run the demo app?
Run the following locally:
```
git clone https://github.com/dscarb21/netflix-nlp.git
cd netflix-nlp/
pip install Flask
python3 app.py
```
The Python file should run and display an IP address that can be used to demo the application on a web browser.

## References and Resources
We didn't go through Natural Language Processing in class. Therefore, we relied on other educational resources to navigate the intricacies of our project's scope, especially in regards to learning about NLP techniques. Our primary resource was [DataCamp](https://www.datacamp.com) projects and NLTK course. We also read the documentation of libraries used, such as SKLearn, Pandas, NumPy, NLTK, and Matplotlib to achieve our the results.
