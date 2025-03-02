# Text-Mining
For this project I use a text corpus of twitter posts (tweets) with an emotion class label for each respective text document. The goal is to generate my own emotion based corpus using n-gram Maximum Likelihood Estimation and to then predict its emotion class labels. For the classification I have used the Bag-of-Words method  preprocess a document for a classifier to learn from. 

Separately, I trained a Word2Vec model based on skip-grams to learn contexts of words and embed them into vectors, so that I could explore some cosine similarities.

# Tools used:
**Languages** Python
**Libraries** NLTK, Gensim, Scikit-learn, Pandas, Numpy, Matplotlib, Seaborn

# Key Features:
1. **Data Preprocessing** Tokenization, stopword removal, lemmatization.  
2. **Text Vectorization** Bag of Words model for converting text to vectors.  
3. **Model Training** SGDClassifier for emotion prediction.  
4. **N-gram Generation** 3-gram and 5-gram models to create new documents.  
5. **Evaluation** Accuracy and class distribution comparison.

6. To actually train the model, I split the given 'emotions.csv' into a training and test corpus where 80% of the dataset is used for training and the remaining 20% for testing.
