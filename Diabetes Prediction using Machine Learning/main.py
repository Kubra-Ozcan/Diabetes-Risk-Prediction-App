import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("C:/Users/kubra/OneDrive/Masaüstü/diabetes.csv")

# Make a copy of the dataset
df = data.copy()

# Separate features and target variable
X = df.drop(columns="Outcome", axis=1)
y = df["Outcome"]

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling for better model performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Function to train model, make predictions, and calculate accuracy
def evaluate_model(model):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return round(accuracy * 100, 2)

# List of models to evaluate
models = [
    ("Logistic Regression", LogisticRegression(random_state=0)),
    ("K-Nearest Neighbors", KNeighborsClassifier()),
    ("Support Vector Classifier", SVC(random_state=0)),
    ("Naive Bayes", GaussianNB()),
    ("Decision Tree", DecisionTreeClassifier())
]

# Evaluate each model and store the results
model_names = []
accuracies = []

for name, model in models:
    model_names.append(name)
    accuracies.append(evaluate_model(model))

# Create a DataFrame with model names and their accuracy scores
results = pd.DataFrame(list(zip(model_names, accuracies)), columns=["Model", "Accuracy (%)"])

# Print the results and dataset information
print(results)
print(df.info())
print(data["Pregnancies"])