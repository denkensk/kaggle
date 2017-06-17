# coding:utf-8

import numpy as np
import pandas as pd
import matplotlib as mpl

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
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

    # 从名字中提取称谓
    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # print pd.crosstab(train_df['Title'], train_df['Sex'])

    # 将称谓分类
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

    train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
    test_df = test_df.drop(['Name'], axis=1)
    combine = [train_df, test_df]

    # 对性别进行映射
    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

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

    # 设置AgeBand将年龄映射
    train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
    # print train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

    for dataset in combine:
        dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[dataset['Age'] > 64, 'Age'] = 4

    train_df = train_df.drop(['AgeBand'], axis=1)
    combine = [train_df, test_df]

    # 将是否跟兄弟姐妹、是否跟父母小孩的特征转换为是否Alone
    for dataset in combine:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    # print train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().\
    #     sort_values(by='Survived', ascending=False)

    for dataset in combine:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    # print train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
    train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    combine = [train_df, test_df]

    # # ——————————不太清楚再干嘛
    # for dataset in combine:
    #     dataset['Age*Class'] = dataset.Age * dataset.Pclass
    # # print train_df.loc[:, ['Age*Class', 'Age', 'Pclass']]

    # 挑选出现频率最多的数来填充
    freq_port = train_df.Embarked.dropna().mode()[0]

    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    # print train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean() \
    #     .sort_values(by='Survived', ascending=False)

    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    train_df['Fare'].fillna(train_df['Fare'].dropna().median(), inplace=True)
    test_df['Fare'].fillna(train_df['Fare'].dropna().median(), inplace=True)

    train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
    # print train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean()\
    #     .sort_values(by='FareBand', ascending=True)

    for dataset in combine:
        dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 2
        dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)

    train_df = train_df.drop(['FareBand'], axis=1)
    combine = [train_df, test_df]

    # ——————————————————————分类预测——————————————————————
    X_train = train_df.drop("Survived", axis=1)
    Y_train = train_df["Survived"]
    X_test = test_df.drop("PassengerId", axis=1).copy()

    # ---------------------Logistic 回归------------------
    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    Y_pred = logreg.predict(X_test)
    acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
    print 'Logistic 回归:\t', acc_log

    # ----------------------决策树-------------------------
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, Y_train)
    Y_pred = decision_tree.predict(X_test)
    acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
    print '决策树:\t', acc_decision_tree

    # ----------------------保存结果------------------------
    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })

    submission.to_csv("./output/submission.csv", index=False)











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
