# train.py - trains a Random Forest that predicts a penguin's species
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the Palmer Penguins dataset
df = pd.read_csv("data/penguins.csv")
print("Rows before cleaning:", len(df))

# 2. Drop rows that have missing values
df = df.dropna()
print("Rows after cleaning:", len(df))

# 3. Pick the input columns and the target column
features = ["island", "bill_length_mm", "bill_depth_mm",
            "flipper_length_mm", "body_mass_g", "sex"]
X = df[features]
y = df["species"]

# 4. Turn the text columns (island, sex) into 0/1 columns
X = pd.get_dummies(X, columns=["island", "sex"])

# 5. Split into 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 6. Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Check how well it does on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy:", round(accuracy, 3))
print(classification_report(y_test, y_pred))

# 8. Save the model together with the column order it expects
joblib.dump({"model": model, "columns": list(X.columns), "accuracy": accuracy},
            "model.pkl")
print("Saved model.pkl")
