# Text-Mining
For this project I use a text corpus of twitter posts (tweets) with an emotion class label for each respective text document. The goal is to generate my own emotion based corpus using n-gram Maximum Likelihood Estimation and to then predict its emotion class labels. For the classification I have used the Bag-of-Words method  preprocess a document for a classifier to learn from.

First I analyse the corpus to check the label distribution. Then I perform undersampling to make sure I have a corpus with an even class distribution. After, I preprocess the corpus to make it suitable for the Bag-of-Words model. To actually train the model, I split the given 'emotions.csv' into a training and test corpus where 80% of the dataset is used for training and the remaining 20% for testing.

Then I preprocess the corpus once again to make it suitable for the n-gram Maximum Likelihood Estimation model and produce 2 new corpuses with 1000 documents each with a 2-gram and 5-gram trained models on the input 'emotions.csv' corpus. I then check which one of the generated corpuses has a more balanced class distribution.

Separately, I trained a Word2Vec model based on skip-grams to learn contexts of words and embed them into vectors, so that I could explore some cosine similarities.

# Tools used:
- **Languages:** Python
- **Libraries:** NLTK, Gensim, Scikit-learn, Pandas, Numpy, Matplotlib, Seaborn

# Key Features:
1. **Data Preprocessing** Tokenization, stopword removal, lemmatization.  
2. **Text Vectorization** Bag of Words model for converting text to vectors.  
3. **Model Training** SGDClassifier for emotion prediction.  
4. **N-gram Generation** 3-gram and 5-gram models to create new documents.  
5. **Evaluation** Accuracy and class distribution comparison.

# How to set up the project using Git Bash:
```bash
git clone https://github.com/SlaviMandazhiev/Text-Mining.git
cd Text-Mining

