# Natural Language Processing
import nltk
nltk.download('stopwords')

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter='\t', quoting=3)

# Cleaning the texts
import re
# import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = [] 
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Training the Decision Tree model
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Evaluating the model
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

ac = accuracy_score(y_test, y_pred)
print("Accuracy:", ac)

bias = classifier.score(X_train, y_train)
print("Bias:", bias)

variance = classifier.score(X_test, y_test)
print("Variance:", variance)

# Saving the model
import pickle
model_filename = "sentiment_analysis_model.pkl"
with open(model_filename, 'wb') as model_file:
    pickle.dump(classifier, model_file)

# Saving the vectorizer
vectorizer_filename = "tfidf_vectorizer.pkl"
with open(vectorizer_filename, 'wb') as vectorizer_file:
    pickle.dump(cv, vectorizer_file)

print("Model and vectorizer saved successfully.")
