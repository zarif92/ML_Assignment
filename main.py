from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import nest_asyncio
import uvicorn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the KNN model
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

# Train the Random Forest Classifier model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Train the SVM Classifier model
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Train the Logistic Regression Classifier model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Evaluate the models
knn_accuracy = accuracy_score(y_test, knn_model.predict(X_test))
rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
svm_accuracy = accuracy_score(y_test, svm_model.predict(X_test))
lr_accuracy = accuracy_score(y_test, lr_model.predict(X_test))

print("KNN Accuracy:", knn_accuracy)
print("Random Forest Accuracy:", rf_accuracy)
print("SVM Accuracy:", svm_accuracy)
print("Logistic Regression Accuracy:", lr_accuracy)

# Save the trained models
joblib.dump(knn_model, 'knn_model.joblib')
joblib.dump(rf_model, 'rf_model.joblib')
joblib.dump(svm_model, 'svm_model.joblib')
joblib.dump(lr_model, 'lr_model.joblib')

# Create FastAPI app
app = FastAPI()

# Define the input data model
class TargetData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Load the trained models
knn_model = joblib.load("knn_model.joblib")
rf_model = joblib.load("rf_model.joblib")
svm_model = joblib.load("svm_model.joblib")
lr_model = joblib.load("lr_model.joblib")

# Create an endpoint for KNN model
@app.post("/predict/knn")
def predict_knn(data: TargetData):
    input_data = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    prediction = knn_model.predict(input_data)
    return {"prediction": prediction[0]}

# Create an endpoint for Random Forest model
@app.post("/predict/rf")
def predict_rf(data: TargetData):
    input_data = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    prediction = rf_model.predict(input_data)
    return {"prediction": prediction[0]}

# Create an endpoint for SVM model
@app.post("/predict/svm")
def predict_svm(data: TargetData):
    input_data = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    prediction = svm_model.predict(input_data)
    return {"prediction": prediction[0]}

# Create an endpoint for Logistic Regression model
@app.post("/predict/lr")
def predict_lr(data: TargetData):
    input_data = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    prediction = lr_model.predict(input_data)
    return {"prediction": prediction[0]}
if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run(app, host="127.0.0.1", port=8000)




