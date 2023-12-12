import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

dataset = pd.read_csv('spam.csv')
dataset.head(10)

messages = dataset["EmailText"].tolist()
output_labels = dataset["Label"].values

processed_messages = []

for message in messages:
    message = re.sub(r'\W', ' ', message)
    message = re.sub(r'\s+[a-zA-Z]\s+', ' ', message)
    message = re.sub(r'\^[a-zA-Z]\s+', ' ', message)
    message = re.sub(r'\s+', ' ', message, flags=re.I)
    message = re.sub(r'^b\s+', '', message)
    processed_messages.append(message)

X_train, X_test, y_train, y_test = train_test_split(processed_messages,
                                                    output_labels,
                                                    test_size=0.2,
                                                    random_state=0)

vectorizer = TfidfVectorizer(max_features=2000,
                             min_df=5,
                             max_df=0.75,
                             stop_words=stopwords.words('english'))
X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()

spam_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
spam_classifier.fit(X_train, y_train)

y_pred = spam_classifier.predict(X_test)

print(accuracy_score(y_test, y_pred))
