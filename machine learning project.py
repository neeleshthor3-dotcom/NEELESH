import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Sample dataset (replace with real data)
data = {
    "text": [
        "Verify your account immediately",
        "Meeting scheduled for tomorrow",
        "Click here to reset your password",
        "Lunch plans today?",
        "Urgent wire transfer required"
    ],
    "label": [1, 0, 1, 0, 1]  # 1 = fraud, 0 = safe
}

df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("model", LogisticRegression())
])

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)
print(classification_report(y_test, predictions, zero_division=0))
