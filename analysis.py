import pandas as pd

df = pd.read_csv("data/ai4i2020.csv")

print("Dataset Shape:", df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nMachine Failure Distribution:")
print(df["Machine failure"].value_counts())

# Feature Engineering

df["temp_difference"] = (
    df["Process temperature [K]"]
    - df["Air temperature [K]"]
)

df["torque_speed_ratio"] = (
    df["Torque [Nm]"]
    / df["Rotational speed [rpm]"]
)

print("\nNew Features Created Successfully")

print("\nMachine Failure Distribution:")
print(df["Machine failure"].value_counts())

df["temp_difference"] = (
    df["Process temperature [K]"]
    - df["Air temperature [K]"]
)

df["torque_speed_ratio"] = (
    df["Torque [Nm]"]
    / df["Rotational speed [rpm]"]
)

print("\nFeature Engineering Completed")

import numpy as np

df["ambient_temperature"] = np.random.normal(30, 5, len(df))
df["load_density"] = np.random.uniform(0.3, 1.0, len(df))

print("Contextual features added")

# Correlation with target

corr = df.corr(numeric_only=True)

print("\nTop Features Related to Machine Failure:")
print(
    corr["Machine failure"]
    .sort_values(ascending=False)
    .head(10)
)

import matplotlib.pyplot as plt

failure_counts = df["Machine failure"].value_counts()

failure_counts.plot(kind="bar")

plt.title("Machine Failure Distribution")
plt.xlabel("Failure")
plt.ylabel("Count")

plt.savefig("failure_distribution.png")

print("Graph saved successfully")

df.to_csv("processed_data.csv", index=False)
print("Processed dataset saved")

import matplotlib.pyplot as plt

df["Machine failure"].value_counts().plot(kind="bar")
plt.savefig("failure_distribution.png")

print("Project progress: Week 1 and Week 2 tasks completed")

print("Data fusion pipeline ready for modeling stage")

from sklearn.model_selection import train_test_split

X = df.drop(columns=["Machine failure"])
X = pd.get_dummies(X)

y = df["Machine failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))


from sklearn.metrics import classification_report

print(classification_report(y_test, pred))


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, pred)

print(cm)