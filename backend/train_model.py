import pandas as pd
import numpy as np
import re
import string
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Load datasets
true_news = pd.read_csv('True.csv', low_memory=False)
fake_news = pd.read_csv('Fake.csv', low_memory=False)

# Assign labels
true_news['label'] = 1  # Real news
fake_news['label'] = 0  # Fake news

# Combine datasets
data = pd.concat([true_news, fake_news])

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    return text

# Apply preprocessing
data['text'] = data['title'] + " " + data['text']
data['text'] = data['text'].apply(preprocess_text)

# Splitting data
X = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Handling imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_vec, y_train)

# Train Scikit-learn models
models = {
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier()
}

for name, model in models.items():
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test_vec)
    print(f"\n{name} Model Evaluation:")
    print(classification_report(y_test, y_pred))
    
    # Save model as .pkl
    joblib.dump(model, f"{name.replace(' ', '_')}.pkl")

# Save vectorizer
joblib.dump(vectorizer, "vectorizer.pkl")

# PyTorch Model - Simple Logistic Regression Equivalent
class FakeNewsClassifier(nn.Module):
    def __init__(self, input_dim):
        super(FakeNewsClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, 2)  # Binary classification

    def forward(self, x):
        return self.fc(x)

# Train PyTorch model
input_dim = X_train_vec.shape[1]
pytorch_model = FakeNewsClassifier(input_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(pytorch_model.parameters(), lr=0.01)

X_train_tensor = torch.tensor(X_train_vec.toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = pytorch_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Save PyTorch model
torch.save(pytorch_model.state_dict(), "fake_news_model.pth")
torch.save(vectorizer, "vectorizer.pth")
