# make predictions
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
#getting input and formatting them
inputs=[[]]
inputs[0].append(round(float(input("enter sepal lenth in centimiters: ")),1))
inputs[0].append(round(float(input("enter sepal width: ")),1))
inputs[0].append(round(float(input("enter petal lenth: ")),1))
inputs[0].append(round(float(input("enter petal width: ")),1))
# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
# Split-out validation dataset
array = dataset.values
X_train = array[:,0:4]
y_train = array[:,4]
# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, y_train)
predictions = model.predict(inputs)
# Evaluate predictions
print("your flower is an",predictions[-1])
