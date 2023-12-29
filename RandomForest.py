import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import string
import re
from nltk.corpus import stopwords


def load_dataset(file_path):
    dataset = pd.read_csv(file_path)
    print("READ DONE", dataset.shape)
    return dataset


def process_text(text):
    text = re.sub('\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr', text)
    text = re.sub('(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr', text)
    text = re.sub('£|\$', 'moneysymb', text)
    text = re.sub(
        '\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
        'phonenumbr', text)
    text = re.sub('\d+(\.\d+)?', 'numbr', text)

    text = re.sub('[^\w\d\s]', ' ', text)

    text = text.lower()

    text = text.split()
    return ' '.join(text)


def main():
    dataset = load_dataset('spam_added.csv')

    dataset["ProcessedText"] = dataset["EmailText"].apply(process_text)

    X_train, X_test, y_train, y_test = train_test_split(
        dataset["ProcessedText"],
        dataset["Label"].values,
        test_size=0.2,
        random_state=45,
        stratify=dataset["Label"])

    # Display class distribution in train and test sets
    print("Train Set Class Distribution:",
          np.sum(y_train == "ham") / np.sum(y_train == "spam"))
    print("Test Set Class Distribution:",
          np.sum(y_test == "ham") / np.sum(y_test == "spam"))
    print("TEST PROCESING:\n", process_text(dataset["EmailText"].tolist()[2]))

    # TF-IDF Vectorization and Random Forest Classifier in a Pipeline
    pipeline = Pipeline([('tfidf',
                          TfidfVectorizer(min_df=5,
                                          max_df=0.75,
                                          preprocessor=None,
                                          max_features=1000)),
                         ('rf',
                          RandomForestClassifier(max_depth=None,
                                                 n_estimators=150,
                                                 random_state=42))])

    # Fit the pipeline on the training data
    pipeline.fit(X_train, y_train)

    # Get the number of features
    num_features = len(pipeline.named_steps['tfidf'].get_feature_names_out())
    print("Number of Features:", num_features)

    # Model Evaluation
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Test Set Accuracy:", accuracy)

    while True:
        msg = input("Input your message:\n")
        if msg.lower() == "quit":
            break

        processed_msg = process_text(msg)
        print("PROCESSED:\n", processed_msg)
        real_time = [processed_msg]
        prediction = pipeline.predict(real_time)

        print("Prediction:", prediction[0])


if __name__ == "__main__":
    main()
