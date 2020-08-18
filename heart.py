#Import
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn .model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns


#Data
data = pd.read_csv('heart.csv')
data.head(10)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]


#Scalling
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
X = scaler.fit_transform(X)


#Splitting
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.33, random_state=18)


#Model
clss = LogisticRegression(random_state=18)
clss.fit(X_train, y_train)
y_pred = clss.predict(X_test)


#Score
print('Train Score: ', clss.score(X_train, y_train))
print('Test Score: ', clss.score(X_test, y_test))
print('Number of Itterations: ', clss.n_iter_)
print('Classes: ', clss.classes_)


#Confusion Matrix
CM = confusion_matrix(y_test, y_pred)
print('Confusion Matrix: \n', CM)
sns.heatmap(CM, center=True)


#Accuracy Score
AccScore = accuracy_score(y_test, y_pred)
print('Accuracy Score: ', AccScore)