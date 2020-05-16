#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import itertools

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import utils


# In[4]:


df=pd.read_csv(r'C:\Users\Admin\Desktop\DelayPred\flights.csv')
df.head()
df['Delayed'] = np.where(df['DepDelay']> 20, 1, 0)

df= df.drop(['ArrDelay', 'DepDelay'], axis=1)

df.head()


# In[5]:


dummy_fields = [ 'Month', 'DayOfWeek', 'Origin', 'Dest', 'Distance', 'CarrierDelay','WeatherDelay','NASDelay','SecurityDelay','LateAircraftDelay']
for each in dummy_fields:
    dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
    df=pd.concat([df, dummies], axis=1)

df.head(10)


# In[6]:


df = df.drop(dummy_fields, axis=1)
#lets setup test and training data
train_size = int(len(df) * .8)
print(train_size)


# In[7]:


data = df[df.columns[:-1]]
labels = df['Delayed']


# In[8]:



labels = utils.to_categorical(labels, 2)


# In[9]:


train_data = data[:train_size]
train_labels = labels[:train_size]
test_data = data[train_size:]
test_labels = labels[train_size:]


# In[10]:


batch_size = 500
epochs=1
train_data.shape


# In[11]:


model=Sequential()
model.add(Dense(512,input_shape=(1163,)))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(2, ))
model.add(Activation('softmax'))


# In[13]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[14]:


history = model.fit(train_data, train_labels,
                   batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1
                   )


# In[15]:


score = model.evaluate(test_data, test_labels, batch_size=batch_size, verbose=1)


# In[16]:


print('Test score: ', score[0])
print('Test accuracy: ', score[1])


# In[17]:


predictions = model.predict(test_data)


# In[26]:


results = np.argmax(predictions, 1)
actual = np.argmax(test_labels, 1)
from sklearn.metrics import classification_report
print('LTSM: ')
print (classification_report(results,actual))


# In[27]:


# True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
TP = np.sum(np.logical_and(results == 1, actual == 1))
 
# True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
TN = np.sum(np.logical_and(results == 0, actual == 0))
 
# False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
FP = np.sum(np.logical_and(results == 1, actual == 0))
 
# False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
FN = np.sum(np.logical_and(results == 0, actual == 1))
 
print('TP: %i, FP: %i, TN: %i, FN: %i' % (TP,FP,TN,FN))

