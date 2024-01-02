import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.tree import plot_tree
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import re
from nltk.corpus import stopwords
import string
import re
from nltk.corpus import stopwords
from sklearn.metrics import precision_recall_fscore_support
import nltk

nltk.download('stopwords')


def visual_Distribution(y_train, y_test):
    # Tính toán tỷ lệ phân bố của nhãn trong X_train và X_test
    train_labels = pd.Series(y_train)
    test_labels = pd.Series(y_test)

    train_label_counts = train_labels.value_counts(normalize=True)
    test_label_counts = test_labels.value_counts(normalize=True)

    # Vẽ biểu đồ hình tròn
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.pie(train_label_counts,
            labels=train_label_counts.index,
            autopct='%1.1f%%',
            startangle=90)
    ax1.set_title('Train Data Label Distribution')

    ax2.pie(test_label_counts,
            labels=test_label_counts.index,
            autopct='%1.1f%%',
            startangle=90)
    ax2.set_title('Test Data Label Distribution')

    plt.show()


def evaluate_and_plot_classification_performance(y_true, y_pred, classes):
    # Ma trận nhầm lẫn
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Đánh giá chi tiết: precision, recall, F1-score
    #F1-score = 2*(precision * recall)/precision + recall
    #   => là một số đo kết hợp giữa precision và recall

    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true, y_pred)

    print("Classification Report:")
    for i, cls in enumerate(classes):
        print(
            f"Class: {cls}, Precision: {precision[i]}, Recall: {recall[i]}, F1-score: {f1_score[i]}"
        )

    # Độ chính xác
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy}")

    # Biểu đồ cột cho precision, recall, F1-score
    plt.figure(figsize=(8, 5))

    bar_width = 0.2
    index = np.arange(len(classes))

    plt.bar(index,
            precision,
            bar_width,
            color='blue',
            alpha=0.7,
            label='Precision')
    plt.bar(index + bar_width,
            recall,
            bar_width,
            color='green',
            alpha=0.7,
            label='Recall')
    plt.bar(index + 2 * bar_width,
            f1_score,
            bar_width,
            color='orange',
            alpha=0.7,
            label='F1-score')

    plt.xlabel('Classes')
    plt.ylabel('Scores')
    plt.title('Precision, Recall, and F1-score for Each Class')
    plt.xticks(index + bar_width, classes)
    plt.legend()
    plt.show()


def evaluate_classification_performance(y_true, y_pred):
    # Ma trận nhầm lẫn
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Đánh giá chi tiết: precision, recall, F1-score
    report = classification_report(y_true, y_pred)
    print("Classification Report:")
    print(report)

    # Độ chính xác
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy}")


def visual_tree(pipeline):
    # Sau khi huấn luyện mô hình Random Forest và vẽ biểu đồ cây đầu tiên

    plt.figure(figsize=(20, 10))
    plot_tree(
        pipeline.named_steps['rf'].estimators_[0],
        filled=True,
        feature_names=pipeline.named_steps['tfidf'].get_feature_names_out())
    plt.show()


def load_dataset(file_path):
    dataset = pd.read_csv(file_path)
    print("READ DONE", dataset.shape)
    return dataset


def process_text(text: str):
    text = text.lower().replace('åÕ', "").replace('‰ûò', ' ').replace(
        'ì_', ' ').replace('å£', " £").replace('ìï', " ").replace(
            '‰û_', " ").replace('åè', " ").replace('‰ûï', " ").replace(
                'åð', " ").replace('ìä', " ").replace('åô', " ").replace(
                    'åò', " ").replace('‰ûª', " ")
    text = re.sub('\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr', text)
    text = re.sub('(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr', text)
    text = re.sub('£|\$', 'moneysymb', text)
    text = re.sub(
        '\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
        'phonenumbr', text)
    text = re.sub('\d+(\.\d+)?', 'numbr', text)

    text = re.sub('[^\w\d\s]', ' ', text)

    text = text.split()
    return ' '.join(text)


def main():
    dataset = load_dataset('spam_added.csv')

    dataset["ProcessedText"] = dataset["EmailText"].apply(process_text)

    X_train, X_test, y_train, y_test = train_test_split(
        dataset["ProcessedText"],
        dataset["Label"].values,
        test_size=0.2,
        random_state=98,
        stratify=dataset["Label"])

    print("Train Set Class Distribution:",
          np.sum(y_train == "ham") / np.sum(y_train == "spam"))
    print("Test Set Class Distribution:",
          np.sum(y_test == "ham") / np.sum(y_test == "spam"))
    # print("TEST PROCESING:\n", process_text(dataset["EmailText"].tolist()[2]))

    pipeline = Pipeline([
        ('tfidf',
         TfidfVectorizer(max_features=1000,
                         stop_words=stopwords.words('english'))),
        ('rf',
         RandomForestClassifier(max_depth=None,
                                n_estimators=150,
                                random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    num_features = len(pipeline.named_steps['tfidf'].get_feature_names_out())
    print("Number of Features:", num_features)

    # vocabulary_mapping = pipeline.named_steps['tfidf'].vocabulary_
    # sorted_vocabulary = sorted(vocabulary_mapping.items(), key=lambda x: x[1])

    # for word, index in sorted_vocabulary:
    #     print(f"Index: {index}, Word: {word}")

    y_pred = pipeline.predict(X_test)

    print("Test Set Class Distribution:", np.sum(y_test == "ham"), '/',
          np.sum(y_test == "spam"))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    false_positive_indices = np.where((y_test == 'spam')
                                      & (y_pred == 'ham'))[0]

    print("\nMessages Falsely Classified as Ham (Actually Spam):\n")
    for index in false_positive_indices:
        print(
            f"True Label: {y_test[index]}, Predicted Label: {y_pred[index]}, Message: {X_test.iloc[index]}"
        )

    true_negative_indices = np.where((y_test == 'ham') & (y_pred == 'spam'))[0]

    print("\nMessages Falsely Classified as Spam (Actually Ham):\n")
    for index in true_negative_indices:
        print(
            f"True Label: {y_test[index]}, Predicted Label: {y_pred[index]}, Message: {X_test.iloc[index]}"
        )

    accuracy = accuracy_score(y_test, y_pred)
    print("Test Set Accuracy:", accuracy)

    visual_Distribution(y_train, y_test)
    evaluate_and_plot_classification_performance(y_test, y_pred, ['ham', 'spam'])
    evaluate_classification_performance(y_test, y_pred)
    visual_tree(pipeline)

    while True:
        msg = input("Input your message:\n")
        if msg.lower() == "quit":
            break

        processed_msg = process_text(msg)
        print("PROCESSED:\n", processed_msg)
        print("TF-IDF:\n",
              pipeline.named_steps['tfidf'].transform([processed_msg]))
        real_time = [processed_msg]
        prediction = pipeline.predict(real_time)

        print("Prediction:", prediction[0])


if __name__ == "__main__":
    main()
