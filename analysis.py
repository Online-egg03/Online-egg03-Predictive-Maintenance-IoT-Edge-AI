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