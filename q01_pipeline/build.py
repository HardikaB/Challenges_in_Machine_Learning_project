# %load q01_pipeline/build.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
bank = pd.read_csv('data/Bank_data_to_class.csv', sep=',')
label_enc = LabelEncoder()
for column in bank.select_dtypes(include=['object']).columns.values:
    bank[column] = label_enc.fit_transform(bank[column])
X_train, X_test, y_train, y_test = train_test_split(bank.iloc[:,:-1], 
                                                    bank.iloc[:,-1], 
                                                    random_state=9)

rf = RandomForestClassifier(random_state=9)
lr = LogisticRegression(random_state=9)

model=[rf,lr]
# Write your solution here :
def pipeline(X_train, X_test, y_train, y_test,model):
    dict1=dict()
    dataset=[[X_train, X_test, y_train, y_test]]

    # Create the Under samplers
    rus = RandomUnderSampler(random_state=9)
    X_sample2, y_sample2 =  rus.fit_sample(X_train, y_train)
    dataset.append([X_sample2, X_test, y_sample2, y_test])
    
    
    ros = RandomOverSampler(random_state=9)
    X_sample3, y_sample3 = ros.fit_sample(X_train, y_train)
    dataset.append([X_sample3, X_test, y_sample3, y_test])
    
    
    smote = SMOTE(random_state=9, kind='borderline2')
    X_sample4, y_sample4 = smote.fit_sample(X_train, y_train)
    dataset.append([X_sample4, X_test, y_sample4, y_test])
    
    roc_old=0
    roc_new=0
    for m in model:
        for X_train, X_test, y_train, y_test in dataset:
            m.fit(X_train, y_train)
            roc_new=roc_auc_score(y_test, m.predict(X_test))
            if(roc_new>=roc_old):
                dict1.clear()
                dict1[m]=roc_new
                roc_old=roc_new
    return list(dict1.keys())[0],list(dict1.values())[0]

