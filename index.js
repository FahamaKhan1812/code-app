const express = require("express");

const app = express();
const port = 3000;

app.get("/help", (req, res) => {
  res.format({
    text: function () {
      res.send(
        `
	#	-----KNN------
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

#	----NB-----
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
 
  
#  -----BFS----
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
#	----DFS---
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
visited = set()
newArray = []


def dfs(visited, graph, node):
    if node not in visited:
        newArray.append(node)
        visited.add(node)
        for neighbour in graph[node]:
            dfs(visited, graph, neighbour)
        return newArray


print(dfs(visited, graph, 'G'))

# --- Gini ---

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder


dataSet = pd.read_csv('heart.csv')


dataSet.head()


le = LabelEncoder()
df = pd.DataFrame(dataSet)

X = df['Sex'] = le.fit_transform(df['Sex']).reshape(-1, 1)  # 2D-Array
y = df['HeartDisease']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=100)


clf_gini = tree.DecisionTreeClassifier(
    criterion='gini', max_depth=4, min_samples_leaf=4)


# max_depth represents max level allowed in each tree, min_samples_leaf minumum samples storable in leaf node
clf_entropy = tree.DecisionTreeClassifier(
    criterion='entropy', max_depth=4, min_samples_leaf=4)
# fit the tree to iris dataset
clf_gini.fit(X_train, y_train)
clf_entropy.fit(X_train, y_train)


y_pred = clf_gini.predict(X_test)
y_pred


# Function to make predictions
def prediction(X_test, clf_object):

    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print("Predicted values on:")
    print(y_pred)
    return y_pred

# Function to calculate accuracy


def cal_accuracy(y_test, y_pred):

    print("Confusion Matrix: ",
          confusion_matrix(y_test, y_pred))

    print("Accuracy : ",
          accuracy_score(y_test, y_pred))

    print("Report : \n",
          classification_report(y_test, y_pred))


y_pred_gini = prediction(X_test, clf_gini)
# y_pred_entropy = prediction(X_test,clf_entropy)
df_pred = pd.DataFrame()
df_pred['y_test'] = y_test
df_pred['y_pred'] = y_pred_gini
df_pred.head(n=10)

cal_accuracy(y_test, y_pred_gini)

#	---LR---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


data = pd.read_csv('Weather.csv', low_memory = False)

data.columns


deleteColumns = [
        'STA', 'Date', 'Precip', 'WindGustSpd',
       'MeanTemp', 'Snowfall', 'PoorWeather', 'YR', 
       'MO', 'DA', 'PRCP', 'DR',
       'SPD', 'MAX', 'MIN', 'MEA', 'SNF', 'SND', 'FT', 'FB', 'FTI', 'ITH',
       'PGT', 'TSHDSBRSGF', 'SD3', 'RHX', 'RHN', 'RVG', 'WTE'
       ]



data.drop(deleteColumns, axis=1)


data['WindGustSpd'].fillna(1, inplace=True)
data['WindGustSpd']


x = data['MaxTemp'].values.reshape(-1,1)
y = data['MinTemp'].values.reshape(-1,1)

#  [markdown]
# PLotting 2D-Graph


data.plot(x = 'MaxTemp', y ='MinTemp', style= 'x')
plot.title('Max Temp vs Min Temp')
plot.xlabel('Max Temp')
plot.ylabel('Min Temp')

plot.show()


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)


lr = LinearRegression()
lr.fit(x, y)


y_pred = lr.predict(x_test)


df = pd.DataFrame(
    {
        'Actual': y_test.flatten(),
        'Predicted': y_pred.flatten(),
    })

df.head(10)


plot.scatter(x_train, y_train, color = 'red')
plot.plot(x_train, lr.predict(x_train), color = 'blue')
plot.title('Max Vs Min Temperature (Training set)')
plot.xlabel('Max Temp')
plot.ylabel('Min Temp')
plot.show()


plot.scatter(x_test, y_test, color = 'red')
plot.plot(x_test, lr.predict(x_test), color = 'blue')
plot.title('Max Vs Min Temperature (Test set)')
plot.xlabel('Max Temp')
plot.ylabel('Min TEMP')
plot.show()


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


         `
      );
    },
  });
});


app.listen(port, () => {
  console.log(`Example app listening on port ${port}`);
});
