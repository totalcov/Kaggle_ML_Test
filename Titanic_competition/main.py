import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

train = pd.read_csv('Titanic_competition/train.csv')
test = pd.read_csv('Titanic_competition/test.csv')


age_median = train['Age'].median()
fare_median = train['Fare'].median()
embarked_mode = train['Embarked'].mode()[0]

def prepare_data(df):
    df = df.copy()

    df['Age'] = df['Age'].fillna(age_median)
    df['Fare'] = df['Fare'].fillna(fare_median)

    df['Sex_num'] = df['Sex'].replace({'male': 0, 'female': 1}).astype(int)
    df['Embarked'] = df['Embarked'].fillna(embarked_mode)
    df['Embarked_num'] = df['Embarked'].map({
        'S': 0,
        'C': 1, 
        'Q': 2
    }).astype(int)

    return df

train = prepare_data(train)
test = prepare_data(test)


y = train.Survived

main_sign = ['Pclass', 'Sex_num', 'Age', "Embarked_num", 'SibSp', 'Parch', "Fare"]
x = train[main_sign]

test_col_vo = [3,5,7,10,12,15,20,25,30,35,40,45,48,50,55,60,65,70,75,80,85,90,95,100]

best_accuracy = 0
best_leaf_nodes = None

for leaf in test_col_vo:
    model = RandomForestClassifier(max_leaf_nodes=leaf, random_state=1, n_estimators=100)
    score = cross_val_score(model, x, y, cv=5, scoring='accuracy')

    mean_score = score.mean()

    if mean_score > best_accuracy:
        best_accuracy = mean_score
        best_leaf_nodes = leaf

print(best_accuracy, best_leaf_nodes)
#-- реализация самой модели 
main_model = RandomForestClassifier(max_leaf_nodes=best_leaf_nodes, random_state=1, n_estimators=100)
main_model.fit(x, y)


x = test[main_sign]

ret = main_model.predict(x)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": ret
})

submission.to_csv("Titanic_competition/submission.csv", index=False)