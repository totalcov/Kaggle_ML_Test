import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


bd = pd.read_csv('Titanic_competition/test.csv')

print(bd.head())

