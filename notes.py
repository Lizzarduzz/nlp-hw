# Applying Naive Bayes classifier
mnb = MultinomialNB()
mnb.fit(article_train, train['Class'])
article_predict = mnb.predict(article_test)
print('Multinomial Naive Bayes accuracy: ' + str(accuracy_score(article_predict, test['Class'])))