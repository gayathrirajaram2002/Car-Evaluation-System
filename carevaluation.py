# Car Evaluation System using SVM

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"

cols = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
df = pd.read_csv(url, names=cols)

print("First 5 Records:")
print(df.head())

# Encode Categorical Data
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

# Split Features and Target
X = df.drop("class", axis=1)
y = df["class"]

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train SVM Model
model = SVC(kernel='rbf')
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Results
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Sample Prediction
sample = [[0, 0, 2, 2, 2, 2]]
sample = scaler.transform(sample)

prediction = model.predict(sample)

classes = ["acc", "good", "unacc", "vgood"]

print("\nPredicted Car Class:", classes[prediction[0]])
