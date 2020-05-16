#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
np.random.seed(0)
from sklearn.metrics import accuracy_score,recall_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


df=pd.read_csv(r'C:\Users\Admin\Desktop\DelayPred\flights.csv')
df.head()
len(df.columns)
df.head(5)


# In[3]:


dummy_fields = [ 'Month', 'DayOfWeek', 'Origin', 'Dest', 'Distance', 'CarrierDelay','WeatherDelay','NASDelay','SecurityDelay','LateAircraftDelay']
for each in dummy_fields:
    dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
    df=pd.concat([df, dummies], axis=1)

df.head(10)


# In[4]:


df = df.drop(dummy_fields, axis=1)
df['Delayed'] = np.where(df['DepDelay']> 10, 1, 0)

df= df.drop(['ArrDelay', 'DepDelay'], axis=1)
df.head(10)


# In[5]:


def delay_classification(df2, target_var):

    features = df2.columns[0:1163]

    X_1 = np.array(df2[features])

    y_1 = np.array(df2[target_var])

    X_train, X_test, y_train, y_test = train_test_split(X_1, y_1, test_size=0.30, random_state=42)
    print("KNN: ")
    print("no of training samples = ", len(y_train))
    print("no of testing samples = ", len(y_test))



    clf = KNeighborsClassifier(n_neighbors=4)
    clf.fit(X_train, y_train)

    Y_predict = clf.predict(X_test)
    
    print("ACCURACY :     ",accuracy_score(Y_predict, y_test)*100)
    
    print()

    from sklearn.metrics import classification_report 
    print ('Report : ')
    print (classification_report(y_test, Y_predict))

    

    
    
delay_classification(df, 'Delayed')

