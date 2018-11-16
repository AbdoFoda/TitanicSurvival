import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes = True)
train_df = pd.read_csv("./train.csv")
test_df = pd.read_csv("./test.csv")

#fill the missing age with the median
train_df["Age"].fillna(train_df["Age"].median(skipna = True),inplace = True)
#fill the missing embarked with the max value(S)
train_df["Embarked"].fillna(train_df["Embarked"].value_counts().idxmax(),inplace=True)

#ignore cabin,ticket features from training data
train_df.drop("Cabin",axis = 1 , inplace= True)
train_df.drop("Ticket",axis = 1, inplace = True)
train_df.drop("Name",axis = 1 , inplace = True)

#ignore cabin,ticket features from test data
test_df.drop("Cabin",axis = 1 , inplace= True)
test_df.drop("Ticket",axis = 1, inplace = True)
test_df.drop("Name",axis = 1 , inplace = True)



train_df["Sex"] = pd.factorize(train_df["Sex"])[0]
train_df["Embarked"] = pd.factorize(train_df["Embarked"])[0]


test_df["Sex"] = pd.factorize(test_df["Sex"])[0]
test_df["Embarked"] = pd.factorize(test_df["Embarked"])[0]

plt.figure(figsize=(15,8))
ax = sns.kdeplot(train_df["Age"][train_df.Survived == 1], color="darkturquoise", shade=True)
sns.kdeplot(train_df["Age"][train_df.Survived == 0], color="lightcoral", shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Age for Surviving Population and Deceased Population')
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()



colsArray = [
["Age", "Embarked" , "SibSp" ,"Fare" ,"Sex","Parch","Pclass"],
 ["Age", "Embarked" , "SibSp"]
,["SibSp","Fare"]
,["Sex","Pclass"]
,["Fare" ,"Sex" , "Age" , "Parch"]
,["Age" , "SibSp" , "Fare","Sex","Parch","Pclass"]
]
def getAccuracy(cols):
    X = train_df[cols]
    Y = train_df["Survived"]

    X_train, X_test, y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


    model = LogisticRegression()
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    print("Accuracy = %2.3f" % accuracy_score(Y_test,y_pred))
    return accuracy_score(Y_test,y_pred)

for coli in colsArray:
    getAccuracy(coli)


#print(train_df.isnull().sum())




