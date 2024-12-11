#45 Acc with categorical
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

#varnames = ['gender', 'age','weight', 'wc', 'bmi', 'mean_steps', 'std_steps', 'mean_morning_steps', 
#            'mean_afternoon_steps', 'mean_evening_steps', 
#            'mean_calories_burnt', 'std_calories_burnt', 'mean_minutes_rem','std_minutes_rem']
##            'mean_minutes_rem', 'std_minutes_rem']
#dep_varname = 'a1c'
#Y = dataframe[dep_varname]
#def without_keys(d, keys): 
#    return {x: d[x] for x in d if x in keys}
#    return {x: d[x] for x in d if x in keys}
#X = without_keys(dataframe, varnames)
#X = pd.DataFrame.from_dict(X)
##Adding gender as dummy variable
#X_gender = pd.get_dummies(X['gender'])
#X_new = X.drop(labels='gender', axis=1)
#X_new = pd.concat([X_new, X_gender['M']], axis=1)
#yf = Y.astype(float)

#X = X_new
#X = pd.concat([pd.Series(1, index=X.index, name='00'), X], axis=1)
#X = X.astype(float)
X = X.drop('wc',1)
X = X.drop('bmi',1)
X = X.drop('weight',1)
X = X.drop('age',1)
#X = X.drop('gender',1)
X = X.drop('mean_calories_burnt',1)
X = X.drop('std_calories_burnt',1)
X = X.drop('mean_minutes_rem',1)
X = X.drop('std_minutes_rem',1)

#dep_varname = 'tchol'
dep_varname = 'a1c'
y = dataframe[dep_varname]
#y_new = y.astype(float)
#y = y_new
y = y.astype(float)

for i in range(0, len(y)):#range(0, len(y_new)):
    if (y[i]>12.0):
        y = y.drop(i,0)
        X = X.drop(i,0)

y_backup = y#y_new
#for i in range(0, len(y_new)):
#    if y_new[i] < 200:#7:#200:
#        y_class[i] = 0
#    elif y_new[i] <= 240:# 8.4: # 240:
#        y_class[i] = 1
#    elif y_new[i] > 240: #8.4: #240:
#        y_class[i] = 2
        
#for i in range(0, len(y_new)):
#    if y_new[i] < 200:#7:#200:
#        y_class[i] = 0
#    elif y_new[i] <= 240:# 8.4: # 240:
#        y_class[i] = 1
#    elif y_new[i] > 240: #8.4: #240:
#        y_class[i] = 2


#for i in range(0, len(y_new)):
#    if y_new[i] < 7:#200:
#        y_class[i] = 0
#    elif y_new[i] <= 8.4: # 240:
#        y_class[i] = 1
#    elif y_new[i] > 8.4: #240:
#        y_class[i] = 2
   
#Classes
idx_list = list(y.index.values)   
for i in idx_list:#range(0, len(y_new)):
    if y[i] <= 7.5:#200:
        y[i] = 0
    elif y[i] > 7.5: # 240:
        y[i] = 1

#y = y_class


columns = ['mean_morning_steps', 'mean_evening_steps',
       'mean_afternoon_steps', 'mean_steps', 'std_steps']
#columns = ['00', 'mean_morning_steps', 'mean_evening_steps',
#       'mean_afternoon_steps', 'mean_steps', 'std_steps', 'wc', 'bmi',
#       'weight', 'age', 'M']

#for col in columns:
#    X[col] = X[col]/np.max(X[col])
    
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
import pandas as pd
from keras.layers import Dense


y = y.astype(int)
labels = y
features = X

y = np.ravel(labels)
X = features


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


num_features = len(X_train[0])
#np.random.seed(1513)#56acc
np.random.seed(2415)#5946
tf.random.set_seed(3503)  #241,250 - 5882

model = Sequential()
model.add(Dense(8, activation='relu', input_shape=(num_features,)))
model.add(Dense(8, activation='tanh'))
model.add(Dense(1, activation='sigmoid')) #binary
#model.add(Dense(1, activation='softmax'))

#lstm_model.compile(loss='categorical_crossentropy', optimizer=adam_opt,metrics=['accuracy'])

#model.compile(loss='binary_crossentropy',
#              optimizer='sgd',
#              metrics=['accuracy'])

model.compile(loss='BinaryCrossentropy', #'categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
 
#class_weights = {0: 0.63, 1: 0.37}                   
#model.fit(X_train, y_train,epochs=10, batch_size=64, class_weight=class_weights, verbose=1)
                 
#class_weights = {0: 0.63, 1: 0.37}                   
#run 5 tives 100 epochs - Got 61.76 with only 100 epochs
#int(len(y)/2)
model.fit(X_train, y_train,epochs=100, batch_size=64, verbose=1)
#score = model.evaluate(X_test, y_test,verbose=1)

y_pred = model.predict(X_train)
y_pred = model.predict(X_test)
print('TEST ACC')
score = model.evaluate(X_test, y_test,verbose=1)

y_pred_multi = np.argmax(model.predict(X_test), axis=1) #multiclass, uses a `softmax` last-layer activation
y_pred_binary = (model.predict(X_test) > 0.5).astype("int32") #binary, uses a `sigmoid` last-layer activation
#y_pred_classes = model.predict_classes(X_test) #
#y_pred = model.predict(X_train)
#score = model.evaluate(X_train, y_train,verbose=1)

#print(score)

