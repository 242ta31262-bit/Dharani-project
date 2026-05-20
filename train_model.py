import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

data = pd.read_csv('titanic.csv')

data = data[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Survived']]

data['Age'].fillna(data['Age'].mean(), inplace=True)

data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

X = data[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch']]
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()

model.fit(X_train, y_train)

pickle.dump(model, open('model.pkl', 'wb'))

print("Model Trained Successfully")
