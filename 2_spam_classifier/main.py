"""Spam message classification with TF-IDF + Naive Bayes / Logistic Regression.

Dataset: UCI SMS Spam Collection (5,574 labelled messages, ham or spam).
The same method works for emails: turn text into numbers, then classify.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("data/SMSSpamCollection", sep="\t", header=None,
                 names=["label", "text"])
print(df["label"].value_counts().to_string(), "\n")

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

vec = TfidfVectorizer(lowercase=True, stop_words="english")
Xtr, Xte = vec.fit_transform(X_train), vec.transform(X_test)

for name, model in [("Naive Bayes", MultinomialNB()),
                    ("Logistic Regression", LogisticRegression(max_iter=1000))]:
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    print(f"== {name}: accuracy {accuracy_score(y_test, pred):.3f}")
    print(confusion_matrix(y_test, pred, labels=["ham", "spam"]))
    print(classification_report(y_test, pred, digits=3))

# Try it on new messages
msgs = ["Congratulations! You won a free prize, call now to claim",
        "Are we still meeting for lunch tomorrow?"]
print("Demo:", dict(zip(msgs, model.predict(vec.transform(msgs)))))
