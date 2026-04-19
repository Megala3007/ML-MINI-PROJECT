# -------------------- IMPORT LIBRARIES --------------------
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# -------------------- LOAD DATASET --------------------
data = pd.read_csv("accident_data.csv")

print("Dataset Preview:\n")
print(data.head())

# -------------------- DATA PREPROCESSING --------------------
le = LabelEncoder()

cols = ['weather', 'road_type', 'traffic_density', 'time_of_day', 'visibility']

for col in cols:
    data[col] = le.fit_transform(data[col])

# -------------------- FEATURES & TARGET --------------------
X = data.drop("accident_prone", axis=1)
y = data["accident_prone"]

# -------------------- TRAIN TEST SPLIT --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=7
)

# -------------------- SCALING (IMPORTANT FOR LR) --------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------- MODELS --------------------

# Logistic Regression (weaker model)
lr = LogisticRegression(max_iter=500)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, lr_pred)

# Decision Tree (simple)
dt = DecisionTreeClassifier(max_depth=2, random_state=1)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)

# Random Forest (strong model)
rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# -------------------- ACCURACY --------------------
print("\n=== MODEL ACCURACY ===")
print("Logistic Regression:", lr_acc)
print("Decision Tree:", dt_acc)
print("Random Forest:", rf_acc)

# -------------------- GRAPH: ACCURACY --------------------
models = ["Logistic Regression", "Decision Tree", "Random Forest"]
acc = [lr_acc, dt_acc, rf_acc]

plt.figure()
plt.bar(models, acc)
plt.title("Model Accuracy Comparison")
plt.show()

# -------------------- SIMPLE DECISION TREE --------------------
plt.figure(figsize=(8,5))
plot_tree(dt,
          feature_names=X.columns,
          class_names=["Safe", "Accident"],
          filled=True)
plt.title("Decision Tree (Simple)")
plt.show()

# -------------------- CONFUSION MATRICES --------------------

# Logistic Regression
ConfusionMatrixDisplay(confusion_matrix(y_test, lr_pred)).plot()
plt.title("Logistic Regression")
plt.show()

# Decision Tree
ConfusionMatrixDisplay(confusion_matrix(y_test, dt_pred)).plot()
plt.title("Decision Tree")
plt.show()

# Random Forest
ConfusionMatrixDisplay(confusion_matrix(y_test, rf_pred)).plot()
plt.title("Random Forest")
plt.show()

# -------------------- REPORT --------------------
print("\n=== RANDOM FOREST REPORT ===")
print(classification_report(y_test, rf_pred))