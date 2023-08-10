import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import warnings
from sklearn.exceptions import ConvergenceWarning

nltk.download('punkt')
nltk.download('stopwords')

def tokenize_and_stem(text):
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens if token not in stemmed_stopwords]
    return stemmed_tokens

# Create stemmed stop words
stemmer = PorterStemmer()
stemmed_stopwords = [stemmer.stem(word) for word in stopwords.words('english')]

# Load the dataset
data = pd.read_csv("NewsgroupTopic.csv")
texts = data["text"]
labels = data["Class"]

# Preprocess the text data
vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, stop_words=stemmed_stopwords, ngram_range=(1, 2))
X = vectorizer.fit_transform(texts)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train and evaluate the classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC()
}

# Hyperparameter tuning
params_logreg = {"C": np.logspace(-3, 3, 7), "solver": ["lbfgs", "liblinear"]}
params_svm = {"C": np.logspace(-3, 3, 7)}

for name, clf in classifiers.items():
    if name == "Logistic Regression":
        grid = GridSearchCV(clf, param_grid=params_logreg, cv=5, scoring='accuracy')
        warnings.simplefilter("ignore", category=ConvergenceWarning)
    elif name == "SVM":
        grid = GridSearchCV(clf, param_grid=params_svm, cv=5, scoring='accuracy')
        warnings.simplefilter("ignore", category=ConvergenceWarning)
    else:
        grid = clf
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        
    grid.fit(X_train, y_train)
    best_clf = grid.best_estimator_ if hasattr(grid, 'best_estimator_') else clf
    predictions = best_clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"{name} accuracy: {accuracy * 100:.2f}%")
