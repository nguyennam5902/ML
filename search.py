import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
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
    # Add your regex-based preprocessing steps here
    text = re.sub('\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr', text)
    text = re.sub('(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr', text)
    text = re.sub('£|\$', 'moneysymb', text)
    text = re.sub(
        '\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
        'phonenumbr', text)
    text = re.sub('\d+(\.\d+)?', 'numbr', text)
    # Remove all punctuations
    text = re.sub('[^\w\d\s]', ' ', text)

    # Each word to lower case
    text = text.lower()

    # Splitting words to Tokenize
    text = text.split()
    return ' '.join(text)


def main():
    # Load dataset
    dataset = load_dataset('spam_added.csv')

    # Display class distribution
    label_counts = dataset["Label"].value_counts()
    print("Class Distribution:", label_counts)

    # Preprocess messages
    dataset["ProcessedText"] = dataset["EmailText"].apply(process_text)

    # Train-test split
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

    # TF-IDF Vectorization and Random Forest Classifier in a Pipeline
    pipeline = Pipeline([('tfidf',
                          TfidfVectorizer(min_df=5,
                                          max_df=0.75,
                                          preprocessor=None)),
                         ('rf', RandomForestClassifier())])

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'tfidf__max_features': [1000, 2000, 3000],
        'rf__n_estimators': [50, 100, 150],
        'rf__max_depth': [None, 10, 20],
        'rf__random_state': [42]
    }

    # Use GridSearchCV to find the best parameters
    grid_search = GridSearchCV(pipeline,
                               param_grid,
                               cv=5,
                               scoring='accuracy',
                               verbose=1)
    grid_search.fit(X_train, y_train)

    # Print the best parameters and best accuracy
    print("Best Parameters:", grid_search.best_params_)
    print("Best Accuracy:", grid_search.best_score_)

    # Model Evaluation with best parameters
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Test Set Accuracy with Best Parameters:", accuracy)

    while True:
        msg = input("Input your message:\n")
        if msg.lower() == "quit":
            break

        processed_msg = process_text(msg)
        real_time = [processed_msg]
        prediction = best_model.predict(real_time)

        print("Prediction:", prediction[0])


if __name__ == "__main__":
    main()
