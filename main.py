import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


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

train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.2,  random_state = 0, stratify=y)


def get_best(max_leaf_nodes, train_x, val_x, train_y, val_y):
    titanic_model_train = RandomForestClassifier(max_leaf_nodes=max_leaf_nodes   ,random_state=1, n_estimators=100)
    titanic_model_train.fit(train_x, train_y)
    predictions_val = titanic_model_train.predict(val_x)
    accuracy = accuracy_score(val_y, predictions_val)
    return accuracy

test_col_vo = [5,10,15,20,25,30,35,40,45,50,55,58,60,61,65,70,75,80,85,90,95,100,110,120]


best_accuracy = 0
best_leaf_nodes = None

for i in test_col_vo:
    acc = get_best(i, train_x, val_x, train_y, val_y)
    if acc > best_accuracy:  # ищем максимальную точность, не минимальную
        best_accuracy = acc
        best_leaf_nodes = i
        
print(best_accuracy, best_leaf_nodes)



main_model = RandomForestClassifier(max_leaf_nodes=best_leaf_nodes, random_state=1, n_estimators=100)
main_model.fit(x, y)


x = test[main_sign]

ret = main_model.predict(x)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": ret
})

submission.to_csv("submission.csv", index=False)