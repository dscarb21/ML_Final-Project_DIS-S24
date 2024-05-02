# Analyzing Netflix TV Shows Similarity via Plot Summaries
- **Group Members:** Pedro Cruz, Sydney Eriksson, David Scarborough
- **Professor:** Lucy McCarren
- **Class:** Machine Learning B - DIS Spring 2024

## Introduction
The task of predicting similarity among TV shows holds significant importance as it can enhance the performance of algorithms in delivering personalized media to users. Our project aims to predict the similarity between Netflix shows based on their plots, contributing to the advancement of machine learning in media recommendation systems.

### Project Goal
Can we find similar Netflix shows by analyzing the similarity of show descriptions? 
### Solution Idea
Fit different Netflix shows in some different clusters based on the most important words of their plot descriptions.

## Dataset

> [!NOTE]
> This dataset is from 2021. Using a more up-to-date dataset would be ideal for a more powerful model.

[Netflix TV Shows 2021](https://www.kaggle.com/datasets/muhammadkashif724/netflix-tv-shows-2021) (Source: Kaggle/Muhammad Kashif). Netflix is an American subscription streaming service. The service primarily distributes original and acquired films and television shows from various genres, and it is available internationally in multiple languages. 
- Number of titles: 2662 unique values.
- License: MIT.
- Collaborators: Muhammad Kashif (Owner).

## What is Natural Language Processing?

> [!IMPORTANT]
> This project uses natural language as its primary input data type. Thus, we need to use natural language processing techniques to handle data for this project.

Natural language processing, or NLP, combines computational linguistics—rule-based modeling of human language—with statistical and machine learning models to enable computers and digital devices to recognize, understand, and generate text and speech (IBM).
Several NLP tasks break down human text and voice data in ways that help the computer make sense of what it's ingesting. These tasks include speech recognition, part of speech tagging, word sense disambiguation, translation, etc.

## Project Implementation Steps

- Data preprocessing: Turn natural language data into numerical values that can be understood by a computational model. Methods used:
  - Tokenization & Stemming
  - Filter out unnecessary tokens (punctuation, stopwords, LDA visualization)
  - Filter out widespread words among documents
  - Vectorize words and sentences
- Clustering: Use K-Means Clustering to group shows with most similar plots

 ## Methods

Computers operate on numerical data, meaning they can't grasp textual information directly. Therefore, we need to translate them into numerical representations to enable them to comprehend our text-based plot summaries.

To do so, we used the following methods:

- **Tokenization**: Breaking down articles into individual sentences or words, as needed. We used NLTK's word_tokenize method.
- **Stemming**: Bringing down a word from its different forms to the root word. We used NLTK's SnowballStemmer object.
- **Term Frequency-Inverse Document Frequency (TF-IDF)**: TF-IDF plays a critical role in preprocessing by identifying significant and distinctive words within a document. This process is vital as it guides us in constructing vectors that emphasize the most critical aspects of the sentences. It enables words occurring frequently to exert more significant semantic influence in the vector representation. For instance, when we use TF-IDF in the initial three sentences of "The Wizard of Oz" plot, it highlights 'Toto' as the most significant word, which makes sense considering the movie begins with the protagonist's pet dog. To learn more, we recommend checking out GeeksforGeeks's [Understanding TF-IDF (Term Frequency-Inverse Document Frequency)](https://www.geeksforgeeks.org/understanding-tf-idf-term-frequency-inverse-document-frequency/) article.

After preprocessing, we used a few different methods to analyze the similarity among the shows in our dataset. We used:
- **Clustering K-Means Algorithm**: We used this approach to form clusters to assess similarity among vectors produced during the NLP preprocessing phase.
- **SKLearn's Cosine Similarity**: We can  do some linear algebra to check how "similar" these vectors represent the plots. Cosine similarity measures the similarity between two vectors of an inner product space. It is measured by the cosine of the angle between two vectors and determines whether two vectors are pointing in roughly the same direction. It is often used to calculate document similarity in text analysis (ScienceDirect).

## Learnings & Conclusion

- We used a silhouette score alongside the squared error sum to determine the best value of k to use in K-Means Clustering. The silhouette score measures how close each point is to each other in its group and how far away it is from each point outside of its group. A higher silhouette score is better, and a score between 0.25 and 0.5 is considered fair. We also want to reduce the sum of squared error. A choice of k = 4 seems reasonable since it is the maximum of the silhouette index. Although it is not the minimum of the sum of squared error, the sum of squared error is decreasing relatively slowly given the scale, so that does not weigh as heavily in the decision. 

- Although the method works and is an exciting and intuitive way of categorizing entertainment titles, it's rather simplistic.
- This model doesn't consider the genres of the movies as a feature to determine similarity. However, from empirical experience, we know that this does play a role in clustering. 
- Adding more features to the model could help untie titles and serve as another layer to add complexity to the logic of our model.
- However, we wanted to focus more on the NLP component of this project than anything else. We all expressed similar interests in learning about NLP techniques and unsupervised learning, and focusing our efforts on those techniques was more interesting than trying to replicate a model that most likely already exists and is very straightforward.
- Moving forward, including more features apart from plot descriptions would make the model stronger.
 
## App Demo: Finding Similar Shows 

We've encapsulated our model within a user-friendly front-end application, which allows users to receive recommendations for 5 similar titles based on a user-provided show description. This application leverages the identical model trained on the same dataset outlined in our Jupyter notebook, ensuring consistent results. Please note that recommendations are limited to the 2021 Netflix catalog.

## Ethical Considerations

Today, many apps have customized algorithms for reccomending content to view such as Tiktok, Instagram, Youtube, and Netflix. It is important to note that there can be downsides to having an algorithm only recommend content that you will like. In an extreme case, if someone is consuming a lot of political content related to one specific political group, a "good" algorithm would continue to recommend similar content. This is problematic, because in order for someone to make an informed decision on who to vote for, they need to be exposed to all political parties. If they are only consuming content coming from one opinion, they may vote for someone that they would not have if they were exposed to content from the other side. On a less extreme note, if Netflix only recommends content that is almost exactly the same as what a viewer has been watching, the viewer may miss out on a genre of film that they had never been exposed to.

### How to run the demo app?
Run the following locally:
```
git clone https://github.com/dscarb21/netflix-nlp.git
cd netflix-nlp/
pip install Flask
python3 app.py
```
The Python file should run and display an IP address that can be used to demo the application on a web browser.

## References
> [!NOTE]
> We didn't go through Natural Language Processing in class. Therefore, we relied on other educational resources to navigate the intricacies of our project's scope, especially regarding learning about NLP techniques. Our primary resources were [DataCamp](https://www.datacamp.com) projects and the NLTK course. To achieve our results, we also read the documentation of libraries used, such as SKLearn, Pandas, NumPy, NLTK, and Matplotlib.

- “How Netflix’s Recommendations System Works.” Help Center, help.netflix.com/en/node/100639. Accessed 1 May 2024. 
- Kashif, Muhammad. “Netflix TV Shows 2021.” Kaggle, 4 Apr. 2024, www.kaggle.com/datasets/muhammadkashif724/netflix-tv-shows-2021.
- “What Is Natural Language Processing?” IBM, 19 Mar. 2024, www.ibm.com/topics/natural-language-processing. 
- DataCamp. “Natural Language Processing in Python.” DataCamp, www.datacamp.com/tracks/natural-language-processing-in-python. Accessed 1 May 2024.
- “Scipy.Cluster.Hierarchy.Dendrogram#.” Scipy.Cluster.Hierarchy.Dendrogram - SciPy v1.13.0 Manual, docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html. Accessed 2 May 2024.
- “Cosine Similarity.” Cosine Similarity - an Overview | ScienceDirect Topics, www.sciencedirect.com/topics/computer-science/cosine-similarity. Accessed 2 May 2024. 
