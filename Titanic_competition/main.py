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


    df['Sex_num'] = df['Sex'].replace({'male': 0, 'female': 1}).astype(int)
    df['Embarked'] = df['Embarked'].fillna(embarked_mode)

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

    # Редкие титулы объединяем
    title_map = {
    'Mr': 'Mr',
    'Miss': 'Miss',
    'Mrs': 'Mrs',
    'Master': 'Master',

    'Dr': 'Rare',
    'Rev': 'Rare',
    'Col': 'Rare',
    'Major': 'Rare',
    'Mlle': 'Miss',
    'Countess': 'Rare',
    'Ms': 'Miss',
    'Lady': 'Rare',
    'Jonkheer': 'Rare',
    'Don': 'Rare',
    'Dona': 'Rare',
    'Mme': 'Mrs',
    'Capt': 'Rare',
    'Sir': 'Rare'
    }

    df['Title'] = df['Title'].map(title_map)
    df['Title'] = df['Title'].fillna('Rare')

    df['Age'] = df.groupby(['Title', 'Pclass', 'Sex'])['Age'].transform(
    lambda x: x.fillna(x.median())
)
    # Если где-то все равно остались пропуски
    df['Age'] = df['Age'].fillna(df['Age'].median())
    fare_median = train['Fare'].median()
    df['Fare'] = df['Fare'].fillna(fare_median)
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    
    df = pd.get_dummies(df, columns=['Title', 'Embarked'])
    df['CabinKnown'] = df['Cabin'].notna().astype(int)

    return df



train = prepare_data(train)
test = prepare_data(test)


y = train.Survived

main_sign = ['Pclass', 'Sex_num', 'Age', 'SibSp',  'Parch', "Fare", 'FarePerPerson', 'FamilySize', 'IsAlone','Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Rare', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'CabinKnown']


x = train[main_sign]

test_col_vo = [3,5,7,10,12,15,20,25,30,35,37,40,45,48,50,55,60,65,68,70,73,75,80,85,90,95,100,110,120,130,150,140]

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