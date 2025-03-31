import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load processed data
df = pd.read_csv("D:/vs_code/Customer_Churn_Prediction/data/Customer_churn.csv")

# Preprocess data: Convert non-numeric columns to numeric
for column in df.select_dtypes(include=['object']).columns:
    df[column] = df[column].astype('category').cat.codes

X = df.drop(columns=['Churn'])
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(),
    "XGBoost": XGBClassifier(eval_metric='logloss'),
    "SVM": SVC(kernel="linear", probability=True),
    "Naive Bayes": MultinomialNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Bagging": BaggingClassifier(random_state=42),
    "Extra Trees": ExtraTreesClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

best_model = None
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Model: {name}")
    print(f"Accuracy: {acc}")
    print(classification_report(y_test, y_pred))
    print("=" * 40)

    # Save best model
    if acc > best_accuracy:
        best_model = model
        best_accuracy = acc

# Ensure the models directory exists
output_dir = os.path.abspath("../models")  # Use absolute path
os.makedirs(output_dir, exist_ok=True)

# Save the best model if it exists
if best_model is not None:
    joblib.dump(best_model, os.path.join(output_dir, "best_churn_model.pkl"))
    print("Best model saved:", best_model)
else:
    print("No model achieved better accuracy than the initial threshold.")