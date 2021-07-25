# 98% accuracy
import pandas as pd
from scipy.sparse.construct import random
messages = pd.read_csv('SpamClassifier/SMSSpamCollection',sep='\t',names=['label','message'])
# print(messages.head)

import re
import nltk
# nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

corpus =[]
ps = PorterStemmer()

for i in range(0,len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    # convert in into lower case
    review = review.lower()
    # split the sentences
    review = review.split()
    # Stemming
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # join the word
    review = ' '.join(review)
    # append with corpus
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values


# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)
# print(y_pred)

from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(y_test,y_pred)
print(confusion_m)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)



