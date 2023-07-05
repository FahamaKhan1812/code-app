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
#	----8 Puzzle---

import copy


def move_up(board):
    board_new = copy.deepcopy(board)
    zero_place = board_new.index(0)
    swap_place = zero_place + 3

    if zero_place > 5:
        return board_new
    board_new[zero_place], board_new[swap_place] = board_new[swap_place], board_new[zero_place]
    return board_new


def move_down(board):
    board_new = copy.deepcopy(board)
    zero_place = board_new.index(0)
    swap_place = zero_place - 3

    if zero_place < 3:
        return board_new
    board_new[zero_place], board_new[swap_place] = board_new[swap_place], board_new[zero_place]
    return board_new


def move_right(board):
    board_new = copy.deepcopy(board)
    zero_place = board_new.index(0)
    swap_place = zero_place + 1

    if zero_place == 8 or zero_place == 5 or zero_place == 2:
        return board_new
    board_new[zero_place], board_new[swap_place] = board_new[swap_place], board_new[zero_place]
    return board_new


def move_left(board):
    board_new = copy.deepcopy(board)
    zero_place = board_new.index(0)
    swap_place = zero_place - 1

    if zero_place == 0 or zero_place == 3 or zero_place == 6:
        return board_new
    board_new[zero_place], board_new[swap_place] = board_new[swap_place], board_new[zero_place]
    return board_new


def get_child(board):
    return (move_up(board), move_down(board), move_right(board), move_left(board))


def search(init_board, final_board):
    queue = []
    visited = []
    queue.append(init_board)
    while queue:
        node = queue.pop(0)
        if node == final_board:
            return final_board
        children = get_child(node)
        
        for board in children:
            if board in visited:
                continue
            queue.append(board)


print(search([1, 0, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 6, 7, 8]))

#	----Task 1 Tictac Auto----
import random


def drawBoard(board):
    print('   |   |')
    print(' ' + board[7] + ' | ' + board[8] + ' | ' + board[9])
    print('   |   |')
    print('-----------')
    print('   |   |')
    print(' ' + board[4] + ' | ' + board[5] + ' | ' + board[6])
    print('   |   |')
    print('-----------')
    print('   |   |')
    print(' ' + board[1] + ' | ' + board[2] + ' | ' + board[3])
    print('   |   |')


def inputPlayerLetter():
    return ['X', 'O']


def whoGoesFirst():
    return 'player'


def playAgain():
    print('Do you want to play again? (yes or no)')
    return input().lower().startswith('y')


def makeMove(board, letter, move):
    board[move] = letter


def isWinner(bo, le):
    # Given a board and a player's letter, this function returns True if that player has won.
    # We use bo instead of board and le instead of letter so we don't have to type as much.
    return ((bo[7] == le and bo[8] == le and bo[9] == le) or  # across the top
            (bo[4] == le and bo[5] == le and bo[6] == le) or  # across the middle
            (bo[1] == le and bo[2] == le and bo[3] == le) or  # across the bottom
            (bo[7] == le and bo[4] == le and bo[1] == le) or  # down the left side
            (bo[8] == le and bo[5] == le and bo[2] == le) or  # down the middle
            # down the right side
            (bo[9] == le and bo[6] == le and bo[3] == le) or
            (bo[7] == le and bo[5] == le and bo[3] == le) or  # diagonal
            (bo[9] == le and bo[5] == le and bo[1] == le))  # diagonal


def getBoardCopy(board):
    # Make a duplicate of the board list and return it the duplicate.
    dupeBoard = []

    for i in board:
        dupeBoard.append(i)

    return dupeBoard


def isSpaceFree(board, move):
    return board[move] == ' '


def getPlayerMove(board, computerLetter):
    if computerLetter == 'X':
        playerLetter = 'O'
    else:
        playerLetter = 'X'

    for i in range(1, 10):
        copy = getBoardCopy(board)
        if isSpaceFree(copy, i):
            makeMove(copy, computerLetter, i)
            if isWinner(copy, computerLetter):
                return i

    # Check if the player could win on his next move, and block them.
    for i in range(1, 10):
        copy = getBoardCopy(board)
        if isSpaceFree(copy, i):
            makeMove(copy, playerLetter, i)
            if isWinner(copy, playerLetter):
                return i

    # Try to take one of the corners, if they are free.
    move = chooseRandomMoveFromList(board, [1, 3, 7, 9])
    if move != None:
        return move

    # Try to take the center, if it is free.
    if isSpaceFree(board, 5):
        return 5

    # Move on one of the sides.
    return chooseRandomMoveFromList(board, [2, 4, 6, 8])


def chooseRandomMoveFromList(board, movesList):

    possibleMoves = []
    for i in movesList:
        if isSpaceFree(board, i):
            possibleMoves.append(i)

    if len(possibleMoves) != 0:
        return random.choice(possibleMoves)
    else:
        return None


def getComputerMove(board, computerLetter):
    if computerLetter == 'X':
        playerLetter = 'O'
    else:
        playerLetter = 'X'

    for i in range(1, 10):
        copy = getBoardCopy(board)
        if isSpaceFree(copy, i):
            makeMove(copy, computerLetter, i)
            if isWinner(copy, computerLetter):
                return i

    # Check if the player could win on his next move, and block them.
    for i in range(1, 10):
        copy = getBoardCopy(board)
        if isSpaceFree(copy, i):
            makeMove(copy, playerLetter, i)
            if isWinner(copy, playerLetter):
                return i

    # Try to take one of the corners, if they are free.
    move = chooseRandomMoveFromList(board, [1, 3, 7, 9])
    if move != None:
        return move

    # Try to take the center, if it is free.
    if isSpaceFree(board, 5):
        return 5

    # Move on one of the sides.
    return chooseRandomMoveFromList(board, [2, 4, 6, 8])


def isBoardFull(board):
    # Return True if every space on the board has been taken. Otherwise return False.
    for i in range(1, 10):
        if isSpaceFree(board, i):
            return False
    return True


print('Welcome to Tic Tac Toe!')

while True:
    # Reset the board
    # theBoard = [' '] * 10
    theBoard = [' ', ' ', 'X', 'O', ' ', ' ', 'X', ' ', ' ', 'O']
    playerLetter, computerLetter = inputPlayerLetter()
    turn = whoGoesFirst()
    print('The ' + turn + ' will go first.')
    gameIsPlaying = True

    while gameIsPlaying:
        if turn == 'player':
            # Player's turn.
            drawBoard(theBoard)
            move = getPlayerMove(theBoard, computerLetter)
            makeMove(theBoard, playerLetter, move)

            if isWinner(theBoard, playerLetter):
                drawBoard(theBoard)
                print('Hooray! You have won the game!')
                gameIsPlaying = False
            else:
                if isBoardFull(theBoard):
                    drawBoard(theBoard)
                    print('The game is a tie!')
                    break
                else:
                    turn = 'computer'

        else:
            # Computer's turn.
            move = getComputerMove(theBoard, computerLetter)
            makeMove(theBoard, computerLetter, move)

            if isWinner(theBoard, computerLetter):
                drawBoard(theBoard)
                print('The computer has beaten you! You lose.')
                gameIsPlaying = False
            else:
                if isBoardFull(theBoard):
                    drawBoard(theBoard)
                    print('The game is a tie!')
                    break
                else:
                    turn = 'player'

    if not playAgain():
        break

#	--- Task 2 Multi---
import random

def drawBoard(board):
    print('   |   |')
    print(' ' + board[7] + ' | ' + board[8] + ' | ' + board[9])
    print('   |   |')
    print('-----------')
    print('   |   |')
    print(' ' + board[4] + ' | ' + board[5] + ' | ' + board[6])
    print('   |   |')
    print('-----------')
    print('   |   |')
    print(' ' + board[1] + ' | ' + board[2] + ' | ' + board[3])
    print('   |   |')

def inputPlayerLetter():
    letter = ''
    while not (letter == 'X' or letter == 'O'):
        print('Do you want to be X or O?')
        letter = input().upper()
        print("Player choose: " + letter)
    if letter == 'X':
        return ['X', 'O']
    else:
        return ['O', 'X']

def whoGoesFirst():
    if random.randint(0, 1) == 0:
        return 'computer'
    else:
        return 'player'

def playAgain():
    print('Do you want to play again? (yes or no)')
    return input().lower().startswith('y')

def makeMove(board, letter, move):
    board[move] = letter

def isWinner(bo, le):
    return ((bo[7] == le and bo[8] == le and bo[9] == le) or  # across the top
            (bo[4] == le and bo[5] == le and bo[6] == le) or  # across the middle
            (bo[1] == le and bo[2] == le and bo[3] == le) or  # across the bottom
            (bo[7] == le and bo[4] == le and bo[1] == le) or  # down the left side
            (bo[8] == le and bo[5] == le and bo[2] == le) or  # down the middle
            # down the right side
            (bo[9] == le and bo[6] == le and bo[3] == le) or
            (bo[7] == le and bo[5] == le and bo[3] == le) or  # diagonal
            (bo[9] == le and bo[5] == le and bo[1] == le))  # diagonal

def getBoardCopy(board):
    dupeBoard = []

    for i in board:
        dupeBoard.append(i)

    return dupeBoard

def isSpaceFree(board, move):
    return board[move] == ' '

def getPlayerMove(board):
    move = ' '
    while move not in '1 2 3 4 5 6 7 8 9'.split() or not isSpaceFree(board, int(move)):
        print('What is your next move? (1-9)')
        move = input()
    return int(move)

def chooseRandomMoveFromList(board, movesList):
    possibleMoves = []
    for i in movesList:
        if isSpaceFree(board, i):
            possibleMoves.append(i)

    if len(possibleMoves) != 0:
        return random.choice(possibleMoves)
    else:
        return None

def getComputerMove(board, computerLetter):
    if computerLetter == 'X':
        playerLetter = 'O'
    else:
        playerLetter = 'X'

    for i in range(1, 10):
        copy = getBoardCopy(board)
        if isSpaceFree(copy, i):
            makeMove(copy, computerLetter, i)
            if isWinner(copy, computerLetter):
                return i

    for i in range(1, 10):
        copy = getBoardCopy(board)
        if isSpaceFree(copy, i):
            makeMove(copy, playerLetter, i)
            if isWinner(copy, playerLetter):
                return i

    move = chooseRandomMoveFromList(board, [1, 3, 7, 9])
    if move != None:
        return move

    if isSpaceFree(board, 5):
        return 5

    return chooseRandomMoveFromList(board, [2, 4, 6, 8])

def isBoardFull(board):
    for i in range(1, 10):
        if isSpaceFree(board, i):
            return False
    return True

print('Welcome to Tic Tac Toe!')

while True:
    theBoard = [' ', 'O', 'X', 'O', 'X', ' ', ' ', ' ', ' ', ' ']
    playerLetter, computerLetter = inputPlayerLetter()
    turn = whoGoesFirst()
    print('The ' + turn + ' will go first.')
    gameIsPlaying = True

    while gameIsPlaying:
        if turn == 'player':
            # Player's turn.
            drawBoard(theBoard)
            move = getPlayerMove(theBoard)
            makeMove(theBoard, playerLetter, move)

            if isWinner(theBoard, playerLetter):
                drawBoard(theBoard)
                print('Hooray! You have won the game!')
                gameIsPlaying = False
            else:
                if isBoardFull(theBoard):
                    drawBoard(theBoard)
                    print('The game is a tie!')
                    break
                else:
                    turn = 'computer'

        else:
            move = getComputerMove(theBoard, computerLetter)
            makeMove(theBoard, computerLetter, move)

            if isWinner(theBoard, computerLetter):
                drawBoard(theBoard)
                print('The computer has beaten you! You lose.')
                gameIsPlaying = False
            else:
                if isBoardFull(theBoard):
                    drawBoard(theBoard)
                    print('The game is a tie!')
                    break
                else:
                    turn = 'player'

    if not playAgain():
        print("Thanks for playing!")
        break
#	----ANN-----
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('Weather.csv', low_memory=False)
imputer = SimpleImputer(strategy='median', missing_values=np.nan)
imputer = imputer.fit(df[['MAX', 'MIN']])
df[['MAX', 'MIN']] = imputer.transform(df[['MAX', 'MIN']])


target_column = ['MIN']
predictors = ["MAX"]
df[predictors] = df[predictors]/df[predictors].max()
df.describe().transpose()
df.head()


X = df[predictors].values
y = df[target_column].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=40)


mlp = MLPClassifier(hidden_layer_sizes=(8, 8, 8),
                    activation='logistic', solver='adam', max_iter=10)
mlp.fit(X_train, y_train.ravel())

predict_test = mlp.predict(X_test)


print(confusion_matrix(y_test, predict_test))
print(classification_report(y_test, predict_test))

#	---SVM---
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
data = np.loadtxt('data_banknote_authentication.txt', delimiter=',')

# Split the dataset into features (X) and target variable (y)
X = data[:, :-1]
y = data[:, -1]

# Define the train/test split ratios
split_ratios = [(0.5, 0.5), (0.7, 0.3), (0.9, 0.1)]

# Perform train/test split and evaluate performance for each split ratio
for train_ratio, test_ratio in split_ratios:
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, test_size=test_ratio, random_state=42)
    
    # Initialize and train the SVM classifier
    svm = SVC()
    svm.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    
    print(f"Train/Test Split: {train_ratio}/{test_ratio}")
    print("Confusion Matrix:")
    print(confusion)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("----------------------------------------")

# Perform k-fold cross-validation and evaluate performance
k = 5  # Choose an appropriate value for k
svm_cv = SVC(kernel='linear')
scores = cross_val_score(svm_cv, X, y, cv=k)
mean_accuracy = np.mean(scores)

print(f"Cross-Validation (k={k})")
print("Accuracy scores:", scores)
print("Mean Accuracy:", mean_accuracy)


#	---- Hil Climbing----
import random
import math

# Define the function to be optimized
def f(x, y):
    return math.exp(-(x*2+y*2))

step_size = 0.1

#initial solution
x, y = random.uniform(-5, 5), random.uniform(-5, 5)
score = f(x, y)

#Hill Climbing algorithm
while True:
    # Generate a neighboring solution
    x_new, y_new = x + random.uniform(-step_size, step_size), y + random.uniform(-step_size, step_size)
    score_new = f(x_new, y_new)


    if score_new > score:
        x, y = x_new, y_new
        score = score_new

    else:
        break

# Print the solution and its score
print("x =", x)
print("y =", y)
print("score =", score)

#	---EA---
import random

# Define the genetic algorithm parameters
POPULATION_SIZE = 50
NUM_GENERATIONS = 100
MUTATION_RATE = 0.1

# Define the function to be optimized
def f(x, y):
    return (1-x)*2 + 100(y-x*2)*2

# Define the genotypic representation for the function
def genotypic_representation():
    return [random.uniform(-5, 5), random.uniform(-5, 5)]

# Define the fitness function for the genotypic representation
def fitness(genotype):
    return f(genotype[0], genotype[1])

# Define the mutation operator for the genotypic representation
def mutation(genotype):
    if random.random() < MUTATION_RATE:
        index = random.randint(0, 1)
        genotype[index] += random.uniform(-0.1, 0.1)
    return genotype

# Define the crossover operator for the genotypic representation
def crossover(parent1, parent2):
    child1 = [parent1[0], parent2[1]]
    child2 = [parent2[0], parent1[1]]
    return child1, child2

# Initialize the population with random genotypes
population = [genotypic_representation() for _ in range(POPULATION_SIZE)]

# Run the genetic algorithm for a fixed number of generations
for _ in range(NUM_GENERATIONS):
    # Evaluate the fitness of each individual in the population
    fitness_scores = [fitness(genotype) for genotype in population]

    # Select the parents for mating
    parent_indices = random.choices(range(POPULATION_SIZE), weights=fitness_scores, k=2)
    parent1 = population[parent_indices[0]]
    parent2 = population[parent_indices[1]]

    # Perform crossover and mutation to generate the offspring
    child1, child2 = crossover(parent1, parent2)
    child1 = mutation(child1)
    child2 = mutation(child2)

    # Replace the least fit individuals in the population with the offspring
    population[fitness_scores.index(max(fitness_scores))] = child1
    population[fitness_scores.index(max(fitness_scores))] = child2


fitness_scores = [fitness(genotype) for genotype in population]
best_index = fitness_scores.index(min(fitness_scores))
best_genotype = population[best_index]
best_score = fitness_scores[best_index]

# Print the optimal solution and its score
print("Optimal solution:", best_genotype)
print("Optimal score:", best_score)
         `
      );
    },
  });
});


app.listen(port, () => {
  console.log(`Example app listening on port ${port}`);
});
