import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from sklearn.metrics import accuracy_score
import re
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')

# Load the dataset
data = pd.read_csv("NewsgroupTopic.csv")
texts = data['text']


### Preprocess the text ###
# Case conversion
data['clean_text'] = data['text'].str.lower()

# Tokenization
data['tokenized_text'] = data['clean_text'].apply(nltk.word_tokenize)

# Remove repeated characters
def remove_repeated_chars(tokens):
    return [re.sub(r'(.)\1+', r'\1\1', token) for token in tokens]

data['no_repeated_chars'] = data['tokenized_text'].apply(remove_repeated_chars)

# Remove stopwords
stop_words = set(stopwords.words('english'))
data['filtered_text'] = data['no_repeated_chars'].apply(lambda x: [word for word in x if word not in stop_words])

# Lemmatization
lemmatizer = WordNetLemmatizer()
data['lemmatized_text'] = data['filtered_text'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

# Convert the cleaned text back to a single string
data['cleaned_text'] = data['lemmatized_text'].apply(' '.join)

#Define a smaller dataset for training
train_data = data[['cleaned_text', 'Class']]



### Training and model evaluation ###

# Split the dataset into training and testing sets
train, test = train_test_split(train_data, test_size=0.20, random_state=42)

# Transforming the text to fit the model
tv = TfidfVectorizer()
article_train = tv.fit_transform(train['cleaned_text'])
article_test = tv.transform(test['cleaned_text'])

# Applying Support Vector Machines classifier
from sklearn.svm import LinearSVC
svm = LinearSVC()
svm.fit(article_train, train['Class'])
article_predict = svm.predict(article_test)
print('SVM accuracy: ' + str(accuracy_score(article_predict, test['Class'])))

# Applying Logistic Regression classifier
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(article_train, train['Class'])
article_predict = lr.predict(article_test)
print('Logistic Regression accuracy: ' + str(accuracy_score(article_predict, test['Class'])))

# Applying Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(article_train, train['Class'])
article_predict = rfc.predict(article_test)
print('RFC accuracy: ' + str(accuracy_score(article_predict, test['Class'])))