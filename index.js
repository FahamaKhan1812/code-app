const express = require("express");

const app = express();
const port = 3000;

app.get("/help", (req, res) => {
  res.format({
    text: function () {
      res.send(
        `
		-----KNN------
  	import pandas as pd
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

digits = load_digits()

dir(digits)

df = pd.DataFrame(digits.data, digits.target)

df.head()

df['target'] = digits.target
df.head()

X_train, X_test, y_train, y_test = train_test_split(
    df.drop('target', axis='columns'), df.target, test_size=0.3)

knn = KNeighborsClassifier(n_neighbors=3)


knn.fit(X_train, y_train)
score = knn.score(X_test, y_test)
print(score)

y_pred = knn.predict(X_test)

print(classification_report(y_test, y_pred))

# Load Data from the CSV file
print('-'*55)
print('Loading Data from the CSV file')
iris_df = pd.read_csv(r'iris.csv', names=[
                      'sepal length', 'sepal width', 'petal length', 'petal width', 'target'])

iris_df.replace('Iris-setosa', 0, inplace=True)
iris_df.replace('Iris-versicolor', 1, inplace=True)
iris_df.replace('Iris-virginica', 2, inplace=True)

iris_df

# Split and train the model

X_train, x_test, Y_train, y_test = train_test_split(
    iris_df.drop('target', axis='columns'), iris_df.target, test_size=0.5
)

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_test)
score = knn.score(x_test, y_test)
print(score)

y_pred = knn.predict(x_test)

print(classification_report(y_test, y_pred))

	----N-B-----
 import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

wine = datasets.load_wine()
dir(wine)

df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.3,  random_state=100)

from sklearn.metrics import classification_report
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print("Accuracy Score: ", accuracy_score(y_test, y_pred))
print('-'*65)

le = LabelEncoder()
ds = pd.read_excel("Job_Scheduling.xlsx")

x = ds.iloc[:, 0:2].values
y = ds.iloc[:, 2].values
x[:, 0] = le.fit_transform(x[:, 0])

x[:, 1] = le.fit_transform(x[:, 1])
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

model = GaussianNB()
model.fit(x, y)
y_pred = model.predict(X_test)

print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
print("classification report: \n", classification_report(y_test, y_pred))
print("Predicted Values: ", y_pred)
print("Accuracy Score: ", accuracy_score(y_test, y_pred))
  -----BFS----
  import copy

graph = {
    "A": ["B", "D"],
    "B": ["C", "E"],
    "C": [],
    "D": ["E", "H", "G"],
    "E": ["C", "F"],
    "F": [],
    "G": ["H"],
    "H": []
}

visited_array = []
queue = []


def bfs(graph, start, end):
    queue = []
    queue.append([start])
    while queue:
        path = queue.pop(0)
        node = path[-1]
        if node == end:
            return path
        child = []
        if node in graph:
            child = graph[node]
        for value in child:
            new_path = copy.deepcopy(path)
            new_path.append(value)
            queue.append(new_path)


print(bfs(graph, 'A', 'G'))

         `
      );
    },
  });
});


app.listen(port, () => {
  console.log(`Example app listening on port ${port}`);
});
