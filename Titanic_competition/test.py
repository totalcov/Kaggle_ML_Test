import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.feature_selection import mutual_info_classif
train = pd.read_csv('Titanic_competition/train.csv')
test = pd.read_csv('Titanic_competition/test.csv')

print(train.head())


age_median = train['Age'].median()
fare_median = train['Fare'].median()
embarked_mode = train['Embarked'].mode()[0]



def prepare_data(df):
    df = df.copy()


    df['Sex_num'] = df['Sex'].map({'male': 0, 'female': 1}).astype(int)
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
    
    df['Deck'] = df['Cabin'].str[0]
    df['Deck'] = df['Deck'].fillna('U')
    df['Deck'] = df['Deck'].where(df['Deck'].isin(['A', 'B', 'C', 'D', 'E', 'F', 'G']), 'U')
    
    
    df = pd.get_dummies(df, columns=['Title', 'Embarked', 'Deck'], )
    df['CabinKnown'] = df['Cabin'].notna().astype(int)

    df['TicketGroupSize'] = df.groupby('Ticket')['Ticket'].transform('count')

    return df

train = prepare_data(train)

y = train.Survived

main_sign = ['Pclass', 'Sex_num', 'Age', 'SibSp',  'Parch', "Fare", 'FarePerPerson', 'FamilySize', 'IsAlone','Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Rare', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'CabinKnown', 'Deck_A', 'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E', 'Deck_F', 'Deck_U', 'TicketGroupSize']
x = train[main_sign]


continuous_features = ['Age', 'Fare', 'FarePerPerson']
discrete_features = [col not in continuous_features for col in x.columns]

scr = mutual_info_classif(x, y, discrete_features=discrete_features,random_state=1)


mi_result = pd.DataFrame({
    'feature': x.columns,
    'mi_score': scr
})

mi_result = mi_result.sort_values(by='mi_score', ascending=False)

mi_result['mi_share_%'] = (
    mi_result['mi_score'] / mi_result['mi_score'].sum() * 100
)

print(
    mi_result.to_string(
        index=False,
        formatters={
            'mi_score': '{:.4f}'.format,
            'mi_share_%': '{:.2f}'.format
        }
    )
)


#for i in ['Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Rare']:
#    print(i+  "  -   " + str(train[i].sum()))

