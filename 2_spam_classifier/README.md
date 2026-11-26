# Spam classification

Turns text into TF-IDF numbers, then trains Naive Bayes and Logistic Regression to label messages as ham or spam.

**Data:** UCI SMS Spam Collection (5,574 labelled messages), included in `data/`.
Source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
The method is identical for emails: replace the data file with a two-column (label, text) file.

**Run**
```
pip install -r requirements.txt
python main.py
```
Output: accuracy, confusion matrix, report, and a demo on two new messages (saved example in `run_output.txt`).
