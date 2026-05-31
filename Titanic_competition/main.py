import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

train = pd.read_csv('Titanic_competition/train.csv')
test = pd.read_csv('Titanic_competition/test.csv')


age_median = train['Age'].median()
fare_median = train['Fare'].median()
embarked_mode = train['Embarked'].mode()[0]

def prepare_data(df):
    df = df.copy()


    df['Sex_num'] = df['Sex'].map({'male': 0, 'female': 1}).astype(int)
    df['Embarked'] = df['Embarked'].fillna(embarked_mode)

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    df['FamilyType'] = 'Small'
    df.loc[df['FamilySize'] == 1, 'FamilyType'] = 'Single'
    df.loc[df['FamilySize'] >= 5, 'FamilyType'] = 'Large'



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
    lambda x: x.fillna(x.median()) )



    # Если где-то все равно остались пропуски
    df['Age'] = df['Age'].fillna(df['Age'].median())
    fare_median = train['Fare'].median()
    df['Fare'] = df['Fare'].fillna(fare_median)
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    df['FareLog'] = np.log1p(df['Fare'])
    
    df['Deck'] = df['Cabin'].str[0]
    df['Deck'] = df['Deck'].fillna('U')
    df['Deck'] = df['Deck'].where(df['Deck'].isin(['A', 'B', 'C', 'D', 'E', 'F']), 'U')
    
    
    df = pd.get_dummies(df, columns=['Title', 'Embarked', 'Deck', 'FamilyType'], )
    df['CabinKnown'] = df['Cabin'].notna().astype(int)

    combined = pd.concat([train, test], sort=False)
    combined['TicketGroupSize'] = combined.groupby('Ticket')['Ticket'].transform('count')

    df['IsChild'] = (df['Age'] < 12).astype(int)

    return df



train = prepare_data(train)
test = prepare_data(test)


y = train.Survived

# без 'FamilyType_Single', 'FamilyType_Large', 'FamilyType_Small', 'IsChild'
# без 'SibSp', 'Parch'
main_sign = ['Pclass', 'Sex_num', 'Age', "FareLog", 'FarePerPerson', 'TicketGroupSize',
'               Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Rare', 
            'Embarked_C', 'Embarked_Q', 'Embarked_S',  
            'CabinKnown',
            'FamilySize', 'FamilyType_Single', 'FamilyType_Large', 'FamilyType_Small']


for col in main_sign:
    if col not in train.columns:
        train[col] = 0
    if col not in test.columns:
        test[col] = 0

x = train[main_sign]

test_col_vo = [3, 5, 7, 10, 12, 17, 20,25]
test_min_samples_leaf = [1,3,5,7,9]
test_n_estimators  = [100,150,200]
test_learning_rate = [ 0.05, 0.08, 0.1]


best_accuracy = 0
best_leaf_nodes = None
best_min_samples_leaf = None
best_n_estimators = None
best_learning_rate = None

for leaf in test_col_vo:
    for min_leaf in test_min_samples_leaf:
        for col_tree in test_n_estimators :
            for lr in test_learning_rate:
                model = GradientBoostingClassifier(max_leaf_nodes=leaf, min_samples_leaf=min_leaf, n_estimators = col_tree,
                                                    learning_rate = lr, random_state=1)
                score = cross_val_score(model, x, y, cv=5, scoring='accuracy')

                mean_score = score.mean()

                if mean_score > best_accuracy:
                    best_accuracy = mean_score
                    best_leaf_nodes = leaf
                    best_min_samples_leaf = min_leaf
                    best_n_estimators = col_tree
                    best_learning_rate = lr

print(best_accuracy, best_leaf_nodes, best_min_samples_leaf, best_n_estimators, best_learning_rate)
#-- реализация самой модели 
main_model = GradientBoostingClassifier(max_leaf_nodes=best_leaf_nodes, min_samples_leaf=best_min_samples_leaf,
                                        random_state=1, n_estimators=best_n_estimators, learning_rate=best_learning_rate)
main_model.fit(x, y)


x = test[main_sign]

ret = main_model.predict(x)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": ret
})

submission.to_csv("Titanic_competition/submission.csv", index=False)