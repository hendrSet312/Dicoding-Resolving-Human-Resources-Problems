import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pickle

#mengubah value menjadi interval
def conv_to_interval(label):
    return f'{label.left} - {label.right}'

#mengubah data numerik menjadi kategorik
def conv_to_categorical(df,column,num):
    bins = np.linspace(0,df[column].max(),num=num).round()
    return pd.cut(df[column],bins = bins).apply(lambda x : conv_to_interval(x))

#one hot encoding
def one_hot_encoding(df,column):
    dummies = pd.get_dummies(df[column])
    return pd.concat([df,dummies],axis = 1).drop(columns = [column])


#importing dataset
df = pd.read_csv('employee_data.csv',index_col= 'EmployeeId')

#data cleaning & feature engineering

#menghapus data yang kosong
df.dropna(inplace = True)

#memfilter kolom yang penting
df = df[['JobRole','OverTime','MaritalStatus','RelationshipSatisfaction','EnvironmentSatisfaction','JobLevel',
               'TrainingTimesLastYear','JobSatisfaction','MonthlyRate','HourlyRate','MonthlyIncome','Attrition']].copy()

#feature encoding
df['MaritalStatus'] = df['MaritalStatus'].replace({'Married':1,'Single':-1, 'Divorced':0})
df['OverTime'] = df['OverTime'].apply(lambda x : x == 'Yes')
df['MonthlyRate'] = conv_to_categorical(df,'MonthlyRate',5)
df['HourlyRate'] = conv_to_categorical(df,'HourlyRate',5)
df['MonthlyIncome'] = conv_to_categorical(df,'MonthlyIncome',5)

#one hot encoding di kolom JobRole
categorical_li = ['MonthlyRate','HourlyRate','MonthlyIncome']
for col in categorical_li : 
    df[col] = df[col].cat.codes
df = one_hot_encoding(df,'JobRole')

#memecah dataset menjadi train dan test, dimana dataset train memiliki proporsi 85% dan test 15%
x = df.drop(columns = ['Attrition'])
y = df['Attrition']

x_train,x_test, y_train, y_test = train_test_split(x,y,test_size = 0.15)


#melakukan metode oversampling untuk menyeimbangkan proporsi kolom Attrition
smote = SMOTE()
x_train, y_train = smote.fit_resample(x_train, y_train)

#membuat model prediksi dengan model random forest
rf_model = RandomForestClassifier(random_state=12)
rf_model.fit(x_train,y_train)
y_pred = rf_model.predict(x_test)

filename = 'finalized_model.sav'
pickle.dump(rf_model, open(filename, 'wb'))




