import pandas as pd
import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def normaliser(A):
    for field in dftrm_normal:
        if field in ['Age']:
            dftrm_normal[field] = (dftrm_normal[field]/dftrm_normal[field].abs().max())
        else: 
            continue
    return A

def clean(A):
    #Dropping irrelevant fields
    A= A.drop(['Name','Embarked','Ticket','Cabin', 'Fare', 'SibSp', 'Parch'],axis=1)
    #Removing rows with NA values
    A = A.dropna()
    #Dealing with categorical variables
    A['Sex'] = A['Sex'].replace({'male': 1, 'female': 0})
    return A

def crinptr(A):
    X = A[['Sex', 'Age', 'Pclass']].to_numpy()
    Y = A[['Survived']].to_numpy()
    return X,Y

def crinpts(A):
    X_test = A[['Sex', 'Age', 'Pclass']].to_numpy()
    Y_test = []
    return X_test, Y_test

def plotdes(A,B,C,D):
    #Scatter
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(A, B, C, c = D, marker = '^', cmap = "nipy_spectral")
    plt.title('Survivability in the titanic')
    ax.set_xlabel('Sex');ax.set_ylabel('Age');ax.set_zlabel('Pclass')
    #Show
    plt.show()
    
#TRAINING
#Reading in training data
hndl = open("titanic/train.csv")
dftr = pd.read_csv(hndl)
#Cleaning testing data
dftrm_normal = dftr.copy()
dftrm_normal = clean(dftrm_normal)
#Normalising
dftrm_normal = normaliser(dftrm_normal)
#Creating input and output fields
X, Y = crinptr(dftrm_normal)
#Splitting into testing and training data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#Classifier
clf = RandomForestClassifier()
clf.fit(X_train,Y_train.ravel())

#TESTING
Y_pred = (clf.predict(X_test))
#PLOT
#plotdes((X_test[:,0]), (X_test[:,1]), (X_test[:,2]), Y_test)

#Output data
#Reading in answer data
hndl = open("titanic/test.csv")
dfts = pd.read_csv(hndl)
#Cleaning answer data and making it fit format
dfts_normal = dfts.copy()
dfts_normal = clean(dfts_normal)
#Normalising
dfts_normal = normaliser(dfts_normal)
#Creating answer input fields
X_test, Y_test = crinpts(dfts_normal)
#Storing output
Y_test = (clf.predict(X_test))


