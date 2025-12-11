# titanic_pipeline.py
# Beginner-friendly Titanic pipeline for CodSoft Task 1
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# Ensure screenshots folder exists
os.makedirs("screenshots", exist_ok=True)

print("Loading train.csv (put train.csv in the same folder before running)...")
train = pd.read_csv("train.csv")   # local file required

# Quick info
print("\nDataset shape:", train.shape)
print(train.isnull().sum())

# Preprocessing function
def preprocess(df, fit_imputers=None):
    df = df.copy()
    # keep simple, useful columns
    df = df[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked',]]
    if fit_imputers is None:
        age_imp = SimpleImputer(strategy='median')
        fare_imp = SimpleImputer(strategy='median')
        emb_imp = SimpleImputer(strategy='most_frequent')
        age_imp.fit(df[['Age']])
        fare_imp.fit(df[['Fare']])
        emb_imp.fit(df[['Embarked']])
        fit_imputers = {'age': age_imp, 'fare': fare_imp, 'emb': emb_imp}
    df['Age'] = fit_imputers['age'].transform(df[['Age']])
    df['Fare'] = fit_imputers['fare'].transform(df[['Fare']])
    df['Embarked'] = fit_imputers['emb'].transform(df[['Embarked']])
    # encode sex
    df['Sex'] = df['Sex'].map({'male':0, 'female':1})
    # one-hot encode Embarked (drop first to avoid dummy trap)
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    # drop PassengerId
    if 'PassengerId' in df.columns:
        df = df.drop(columns=['PassengerId'])
    return df, fit_imputers

# Preprocess
X, imputers = preprocess(train)
y = train['Survived']

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,
                                                  random_state=42, stratify=y)
print("\nTrain/Val shapes:", X_train.shape, X_val.shape)

# Model training
model = RandomForestClassifier(n_estimators=200, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print("\nCV scores:", cv_scores)
print("Mean CV accuracy: {:.4f}".format(cv_scores.mean()))

model.fit(X_train, y_train)

# Evaluate on validation
y_pred = model.predict(X_val)
val_acc = accuracy_score(y_val, y_pred)
print("\nValidation accuracy: {:.4f}".format(val_acc))
print("\nClassification report:\n", classification_report(y_val, y_pred))
cm = confusion_matrix(y_val, y_pred)
print("Confusion matrix:\n", cm)

# Save confusion matrix plot
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("screenshots/confusion_matrix.png")
plt.close()
print("Saved screenshots/confusion_matrix.png")

# Feature importances
feat_imp = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
plt.figure(figsize=(6,4))
feat_imp.plot(kind='bar')
plt.title("Feature Importances")
plt.tight_layout()
plt.savefig("screenshots/feature_importance.png")
plt.close()
print("Saved screenshots/feature_importance.png")

# Optional: prepare test submission if test.csv available
if os.path.exists("test.csv"):
    test = pd.read_csv("test.csv")
    X_test, _ = preprocess(test, fit_imputers=imputers)
    preds = model.predict(X_test)
    submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": preds})
    submission.to_csv("submission.csv", index=False)
    print("Wrote submission.csv")
else:
    print("No test.csv found; skipping submission.")

