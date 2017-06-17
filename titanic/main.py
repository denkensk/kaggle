# coding:utf-8

import numpy as np
import pandas as pd
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    pd.set_option('display.width', 200)

    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    combine = [train_df, test_df]

    # print "Before", train_df.shape, test_df.shape
    train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
    test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
    combine = [train_df, test_df]
    # print "After", train_df.shape, test_df.shape

    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # print pd.crosstab(train_df['Title'], train_df['Sex'])

    for dataset in combine:
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                                     'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    # print train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

    # 对称谓进行映射
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    for dataset in combine:
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)
    # print train_df.head()

    train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
    test_df = test_df.drop(['Name'], axis=1)
    combine = [train_df, test_df]
    # print train_df.shape, test_df.shape

    # 对性别进行映射
    for dateset in combine:
        dateset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0})
    # print combine[0].head()


    # 将age为空的数据，通过相同sex和相同pclass的数据求age的平均值补充
    guess_ages = np.zeros((2, 3))
    for dataset in combine:
        for i in range(0, 2):
            for j in range(0, 3):
                guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j + 1)]['Age'].dropna()
                age_guess = guess_df.median()
                guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

        for i in range(0, 2):
            for j in range(0, 3):
                dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), 'Age'] = \
                guess_ages[i, j]

        dataset['Age'] = dataset['Age'].fillna(0).astype(int)
    # print combine[0].head()


    train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
    print train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)






                #
                # # x = data[range(2, 12)]
                # x = train_df[["PassengerId", "Pclass", "Age", "Sex"]].fillna(0)
                # # y = data["Survived"]
                # # x = data[range(4)]
                # # y = LabelEncoder().fit_transform(data[4])
                # y = LabelEncoder().fit_transform(train_df["Survived"])
                # x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)
                #
                # # print x_train.head()
                # # print y_train.head()
                # model = DecisionTreeClassifier(criterion='entropy')
                # model.fit(x_train, y_train)
                # y_test_hat = model.predict(x_test)
                # print 'accuracy_score:', accuracy_score(y_test, y_test_hat)
