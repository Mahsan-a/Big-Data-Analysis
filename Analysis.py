from datetime import datetime as dt
import csv
import pandas as pd
import datetime as dt
from datetime import datetime
import os
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
#import missingno as msno
from tqdm import tnrange, tqdm_notebook
import tqdm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn import preprocessing
from typing import Tuple, List, Dict, Any, Callable, Union, Iterable
from numpy import dtype, ndarray
import math
from sklearn import tree
import graphviz
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pydot
from sklearn.tree import export_graphviz
from sklearn import preprocessing


master_dict  = dict()

#Add fitbit_sleep
filenames_fitbitActivities = []
def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f
filenames_fitbitActivities = os.listdir('fitbitActivities') 

fitbit_participants = []
for file in filenames_fitbitActivities: 
    if(file=='.DS_Store'):
        continue
    filename = "fitbitActivities/" + file
    df = pd.read_csv(filename)
    
    file = file.replace('.csv','')  #Leave only the subject ID for future use
    fitbit_participants.append(file)
    
    with open(filename, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)        #DictReader
        fieldnames_fitbitSleep = reader.fieldnames #Variables contained
        fieldnames_fitbitSleep = fieldnames_fitbitSleep[:10] #Not including Subject column
        #For each participant 
        #Store all the IDs of subjects with fitbit data for future reference
        subject_ID = file #It is the same ID for all the rows
        #subjects_fitbit.append(subject_ID)
        #Store all the variables corresponding to each participant
        fitbitSleep = dict()
        if subject_ID in master_dict:
            master_dict[subject_ID]['fitbitActivities'] = fitbitSleep
        else:
            master_dict[subject_ID] = {}
            master_dict[subject_ID]['fitbitActivities'] = fitbitSleep
            
        for fieldname in fieldnames_fitbitSleep:
            #Dict['Dict1'] = {}
            master_dict[subject_ID]['fitbitActivities'][fieldname] = df[fieldname].to_dict()
            #append_array.append(row[fieldname])

            
#Add fitbit_sleep
filenames_fitbitSleep = []
filenames_fitbitSleep = os.listdir('fitbitSleep') #name of the subfolder
    
for file in filenames_fitbitSleep: 
    if(file=='.DS_Store'):
        continue
    filename = "fitbitSleep/" + file
    df = pd.read_csv(filename)
    
    file = file.replace('.csv','')  #Leave only the subject ID for future use
    
    
    with open(filename, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)        #DictReader
        fieldnames_fitbitSleep = reader.fieldnames #Variables contained
        fieldnames_fitbitSleep = fieldnames_fitbitSleep[:9] #Not including Subject column
        #For each participant 
        #Store all the IDs of subjects with fitbit data for future reference
        subject_ID = file #It is the same ID for all the rows
        #subjects_fitbit.append(subject_ID)
        #Store all the variables corresponding to each participant
        fitbitSleep = dict()
        if subject_ID in master_dict:
            master_dict[subject_ID]['fitbitSleep'] = fitbitSleep
        else:
            master_dict[subject_ID] = {}
            master_dict[subject_ID]['fitbitSleep'] = fitbitSleep
            
        for fieldname in fieldnames_fitbitSleep:
            #Dict['Dict1'] = {}
            master_dict[subject_ID]['fitbitSleep'][fieldname] = df[fieldname].to_dict()
            #append_array.append(row[fieldname])

    
data_dict = dict()
general_dict = dict()
dictionary = master_dict
participants_IDs = list(dictionary.keys())
metabolism_IDlist = []


#file_name = ‘MF_CE_ Pilot_Metabolism Mahsan.xlsx’
metabolism = pd.read_excel ("metabolism.xlsx")
metabolism = metabolism.values
                               
COL_PARTCIPANTID = 0
COL_DATE = 4
COL_HOUR = 5
COL_NUMBERSTEPS = 35
COL_SED = 25
COL_LIGHT = 26
COL_MV = 27   #33
COL_VIG = 28
COL_WEIGHT = 1
COL_AGE = 2
COL_GENDER = 3
COL_A1C = 10
COL_WAIST = 12
COL_BMI = 17
COL_FBG = 23
COL_TG = 24
COL_TCHol = 25
COL_INSULIN = 32
data_dict['participant'],data_dict['mean_morning_steps'],data_dict['mean_evening_steps'],data_dict['mean_afternoon_steps'] = [],[],[],[]
data_dict['std_morning_steps'],data_dict['std_evening_steps'],data_dict['std_afternoon_steps'],data_dict['mean_steps'] = [],[],[],[]
data_dict['std_steps'],data_dict['mean_calories_burnt'],data_dict['std_calories_burnt'] ,data_dict['mean_minutes_sed']= [],[],[],[]
data_dict['std_minutes_sed'],data_dict['mean_minutes_light_active'],data_dict['std_minutes_light_active'],data_dict['mean_minutes_fair_active']= [],[],[],[]
data_dict['std_minutes_fair_active'],data_dict['mean_minutes_vig_active'],data_dict['std_minutes_vig_active'],data_dict['mean_activity_calories']= [],[],[],[]
data_dict['std_activity_calories'],data_dict['mean_calories_burnt'],data_dict['std_calories_burnt'],data_dict['mean_minutes_sed']= [] ,[],[],[]
data_dict['std_minutes_sed'],data_dict['mean_minutes_light_active'],data_dict['std_minutes_light_active'],data_dict['mean_minutes_fair_active']= [],[],[],[]
data_dict['std_minutes_fair_active'],data_dict['mean_minutes_vig_active'],data_dict['std_minutes_vig_active'],data_dict['mean_activity_calories']= [],[],[],[]
data_dict['std_activity_calories'],data_dict['mean_minutes_sleep'],data_dict['std_minutes_sleep'],data_dict['mean_minutes_sleep'] = [],[],[],[]
data_dict['std_minutes_sleep'],data_dict['mean_minutes_rem'],data_dict['std_minutes_rem'],data_dict['mean_minutes_light']= [] ,[],[],[]
data_dict['std_minutes_light'] ,data_dict['mean_minutes_deep'],data_dict['std_minutes_deep'],data_dict['mean_minutes_inbed']= [],[],[],[]
data_dict['std_minutes_inbed'],data_dict['mean_number_awakenings'],data_dict['std_number_awakenings'] = [],[],[]
data_dict['a1c'],data_dict['wc'],data_dict['bmi'],data_dict['weight'],data_dict['age'],data_dict['gender'] = [],[],[],[],[],[]
data_dict['fbg'],data_dict['tg'],data_dict['tchol'],data_dict['insulin'] = [], [], [], []
data_dict['mean_acti_sed'],data_dict['std_acti_sed'],data_dict['mean_acti_light_active'],data_dict['std_acti_light_active']= [],[],[],[]
data_dict['mean_acti_mv_active'],data_dict['std_acti_mv_active'],data_dict['mean_acti_vig_active'],data_dict['std_acti_vig_active']= [],[],[],[]

HourlyData = pd.read_excel (r'HourlyDetailed.xlsx')
participantID = HourlyData.iloc[:,COL_PARTCIPANTID]
participantID = participantID.astype(str)
[participantID_unique,ia,ib] = np.unique(participantID, return_index=True, return_inverse=True)
HourlyData = HourlyData.values
ia = sorted(ia)
ia = np.append(ia,(len(participantID)))

#Assuming a dictionary with ‘actigraph’ ‘Fitbit_activities’ and fitbit_sleep' data for each participant
for idx_participant in range (0,len(participantID_unique)):
    actigraph = []
    morning_steps = []
    afternoon_steps = []
    evening_steps = []
    listofpoints = []
    #HourlyData_currparticipant = HourlyData[(ia[idx_participant]):ia[idx_participant+1],:];
    actigraph = HourlyData[(ia[idx_participant]):ia[idx_participant+1],:]
    participant = actigraph[0,0]
    data_dict['participant'].append(participant)
    date = actigraph[: ,COL_DATE];
        
    #Extracting actigraph values
    #Assuming actigraph is the full hourly information on one participant
    [date_unique,ix,iy] = np.unique(date, return_index=True, return_inverse=True); 
    ix=sorted(ix)
    ix = np.append(ix,(len(actigraph)))
    number_steps_eachday = []
    vig_eachday = []
    sed_eachday = []
    light_eachday = []
    mv_eachday = []
    #separating data by date
    actigraph = np.array(actigraph) 
    date = actigraph[: ,COL_DATE];
    [date_unique,ix,iy] = np.unique(date, return_index=True, return_inverse=True); # unique(datestr(date{:,1}),'stable');
    ix=sorted(ix)
    ix = np.append(ix,(len(actigraph)))
    for idx in range (1,len(date_unique)-1):
        actigraph_eachday = actigraph[(ix[idx]):ix[idx+1],:];
        datetime_eachday = []
        avgsteps_eachday = []
        #reading data for each day
        for i in range (0,len(actigraph_eachday)):
            #time = str(actigraph_eachday[i,COL_HOUR])
            #in_time = dt.strptime(time, "%I:%M %p")
            #out_time = dt.strftime(in_time, "%H:%M")
            out_time = actigraph_eachday[i,COL_HOUR].strftime("%H:%M")
            value= actigraph_eachday[i,COL_NUMBERSTEPS]
            datetime_eachday = np.append(datetime_eachday,(out_time))
            avgsteps_eachday = np.append(avgsteps_eachday,(value))

        #summing up the number of steps each day
        values1 = np.sum(actigraph_eachday[:,COL_NUMBERSTEPS])
        number_steps_eachday = np.append(number_steps_eachday ,(values1))
        values2 = np.sum(actigraph_eachday[i,COL_SED])
        sed_eachday = np.append(sed_eachday,(values2))
        values3 = np.sum(actigraph_eachday[i,COL_LIGHT])
        light_eachday = np.append(light_eachday,(values3))
        values4 = np.sum(actigraph_eachday[i,COL_MV])
        mv_eachday = np.append(mv_eachday,(values4))
        values5 = np.sum(actigraph_eachday[i,COL_VIG])
        vig_eachday = np.append(vig_eachday,(values5))
            
        #appending each day's steps to an array
        t = pd.DataFrame({
            'timestamps': pd.to_datetime(
                datetime_eachday),
            'numsteps':avgsteps_eachday})

        #Getting the 3dimention data for each day 
        #(summing up the number of steps for 6 hours of morning, afternoon and evening)
        t = pd.DataFrame({
            'timestamps': pd.to_datetime(
                datetime_eachday),
            'numsteps':avgsteps_eachday})
        t.index = pd.DatetimeIndex(t['timestamps']).floor('1H')
        all_hours = pd.date_range('00:00', '23:00', freq='1H')
        t=t.reindex(all_hours)
        some_hours = pd.date_range('06:00', '23:00', freq='1H')
        t=t.reindex(some_hours)
        dimention1= pd.date_range('06:00', '11:00', freq='1H')
        t1=t.reindex(dimention1)
        d1= np.sum(t1['numsteps'])
        dimention2= pd.date_range('12:00', '17:00', freq='1H')
        t2=t.reindex(dimention2)
        d2= np.sum(t2['numsteps'])
        dimention3= pd.date_range('18:00', '23:00', freq='1H')
        t3=t.reindex(dimention3)
        d3= np.sum(t3['numsteps'])
        d = [d1,d2,d3]  
        
        # Making a 3d array for each day, now we need to decide how to use it in our model      
        listofpoints.append(d)
        morning_steps.append(d1)
        afternoon_steps.append(d2)
        evening_steps.append(d3)
    
    morning_steps , afternoon_steps = np.array(morning_steps), np.array(afternoon_steps)
    evening_steps, number_steps_eachday = np.array(evening_steps), np.array(number_steps_eachday)
    sed_eachday, light_eachday, mv_eachday, vig_eachday = np.array(sed_eachday), np.array(light_eachday),np.array(mv_eachday), np.array(vig_eachday)
    # we can use the mean and std of morning/afternoon/evening and total number of steps as different feature for our prediction model
    data_dict['mean_morning_steps'].append(np.mean(morning_steps)) # or np.sum(morning_steps)/(len(date_unique)-2) to exclude first and last day of data collection
    data_dict['mean_evening_steps'].append(np.mean(evening_steps))
    data_dict['mean_afternoon_steps'].append(np.mean(afternoon_steps))
    data_dict['std_morning_steps'].append(np.std(morning_steps))
    data_dict['std_evening_steps'].append(np.std(evening_steps))
    data_dict['std_afternoon_steps'].append(np.std(afternoon_steps))
    data_dict['mean_steps'].append(np.mean(number_steps_eachday))
    data_dict['std_steps'].append(np.std(number_steps_eachday))
    data_dict['mean_acti_sed'].append(np.mean(sed_eachday))
    data_dict['std_acti_sed'].append(np.std(sed_eachday))
    data_dict['mean_acti_light_active'].append(np.mean(light_eachday))
    data_dict['std_acti_light_active'].append(np.std(light_eachday))    #for now we can focus on these information from actigraph but we’ll think about other variables to extract from it later
    data_dict['mean_acti_mv_active'].append(np.mean(mv_eachday))
    data_dict['std_acti_mv_active'].append(np.std(mv_eachday))
    data_dict['mean_acti_vig_active'].append(np.mean(vig_eachday))
    data_dict['std_acti_vig_active'].append(np.std(vig_eachday))
    
    if participant in fitbit_participants:
        if 'fitbitActivities' in dictionary[participant]:
            fitbit_activities = dictionary[participant]['fitbitActivities']
            #extracting fitbit activity values
            #It might need a condition here that if fitbit_activities != 'Nan:
            calories_burnt = np.array(list(fitbit_activities['Calories Burned'].items()))[1:-1,1]
            calories_burnt = [float(i.replace(',','')) for i in calories_burnt.tolist()]
            steps = np.array(list(fitbit_activities['Steps'].items()))[1:-1,1]
            steps = [float(i.replace(',','')) for i in steps.tolist()]
            minutes_sed = np.array(list(fitbit_activities['Minutes Sedentary'].items()))[1:-1,1]
            if  ((minutes_sed.dtype != 'int64') & (minutes_sed.dtype != 'int')):
                minutes_sed = [float(i.replace(',','')) for i in minutes_sed.tolist()]
            minutes_light_active = np.array(list(fitbit_activities['Minutes Lightly Active'].items()))[1:-1,1]
            #minutes_light_active = [float(i.replace(',','')) for i in minutes_light_active.tolist()]
            minutes_fair_active = np.array(list(fitbit_activities['Minutes Fairly Active'].items()))[1:-1,1]
            #minutes_fair_active = [float(i.replace(',','')) for i in minutes_fair_active.tolist()]
            minutes_vig_active = np.array(list(fitbit_activities['Minutes Very Active'].items()))[1:-1,1]
            #minutes_vig_active = [float(i.replace(',','')) for i in minutes_vig_active.tolist()]
            activity_calories = np.array(list(fitbit_activities['Activity Calories'].items()))[1:-1,1]
            if  ((activity_calories.dtype != 'int64') & (activity_calories.dtype != 'int')):
                activity_calories = [float(i.replace(',','')) for i in activity_calories.tolist()]
            
            #Adding fitbit activity to final dictionary
            data_dict['mean_calories_burnt'].append(np.mean(calories_burnt)) , data_dict['std_calories_burnt'].append(np.std(calories_burnt))
            data_dict['mean_minutes_sed'].append(np.mean(minutes_sed)), data_dict['std_minutes_sed'].append(np.std(minutes_sed))
            data_dict['mean_minutes_light_active'].append(np.mean(minutes_light_active)), data_dict['std_minutes_light_active'].append(np.std(minutes_light_active))
            data_dict['mean_minutes_fair_active'].append(np.mean(minutes_fair_active)), data_dict['std_minutes_fair_active'].append(np.std(minutes_fair_active))
            data_dict['mean_minutes_vig_active'].append(np.mean(minutes_vig_active)), data_dict['std_minutes_vig_active'].append(np.std(minutes_vig_active))
            data_dict['mean_activity_calories'].append(np.mean(activity_calories)), data_dict['std_activity_calories'].append(np.std(activity_calories))
        else:
            fitbit_activities = float('Nan')
    
        if 'fitbitSleep' in dictionary[participant]:
            fitbit_sleep = dictionary[participant]['fitbitSleep']
            #extracting fitbit sleep values
            #It might need a condition here that if fitbit_activities != 'Nan:
            minutes_sleep = np.array(list(fitbit_sleep['Minutes Asleep'].items()))[1:-1,1]
            minutes_awake = np.array(list(fitbit_sleep['Minutes Awake'].items()))[1:-1,1]
            minutes_rem = np.array(list(fitbit_sleep['Minutes REM Sleep'].items()))[1:-1,1]
            minutes_light = np.array(list(fitbit_sleep['Minutes Light Sleep'].items()))[1:-1,1]
            minutes_deep = np.array(list(fitbit_sleep['Minutes Deep Sleep'].items()))[1:-1,1]
            minutes_inbed = np.array(list(fitbit_sleep['Time in Bed'].items()))[1:-1,1]
            number_awakenings = np.array(list(fitbit_sleep['Number of Awakenings'].items()))[1:-1,1]
        
            #Adding fitbit sleep to final dictionary
            data_dict['mean_minutes_sleep'].append(np.mean(minutes_sleep)) , data_dict['std_minutes_sleep'].append(np.std(minutes_sleep))
            data_dict['mean_minutes_rem'].append(np.mean(minutes_rem)), data_dict['std_minutes_rem'].append(np.std(minutes_rem))
            data_dict['mean_minutes_light'].append(np.mean(minutes_light)), data_dict['std_minutes_light'].append(np.std(minutes_light))
            data_dict['mean_minutes_deep'].append(np.mean(minutes_deep)), data_dict['std_minutes_deep'].append(np.std(minutes_deep))
            data_dict['mean_minutes_inbed'].append(np.mean(minutes_inbed)), data_dict['std_minutes_inbed'].append(np.std(minutes_inbed))
            data_dict['mean_number_awakenings'].append(np.mean(number_awakenings)), data_dict['std_number_awakenings'].append(np.std(number_awakenings))
        else:
            fitbit_sleep = float('Nan')
            data_dict['mean_minutes_sleep'].append(float('Nan')) , data_dict['std_minutes_sleep'].append(float('Nan'))
            data_dict['mean_minutes_rem'].append(float('Nan')), data_dict['std_minutes_rem'].append(float('Nan')),
            data_dict['mean_minutes_light'].append(float('Nan')), data_dict['std_minutes_light'].append(float('Nan'))
            data_dict['mean_minutes_deep'].append(float('Nan')), data_dict['std_minutes_deep'].append(float('Nan')),
            data_dict['mean_minutes_inbed'].append(float('Nan')), data_dict['std_minutes_inbed'].append(float('Nan'))
            data_dict['mean_number_awakenings'].append(float('Nan')), data_dict['std_number_awakenings'].append(float('Nan')) 
    else:
        data_dict['mean_calories_burnt'].append(float('Nan')) , data_dict['std_calories_burnt'].append(float('Nan')) 
        data_dict['mean_minutes_sed'].append(float('Nan')), data_dict['std_minutes_sed'].append(float('Nan'))
        data_dict['mean_minutes_light_active'].append(float('Nan')), data_dict['std_minutes_light_active'].append(float('Nan')) 
        data_dict['mean_minutes_fair_active'].append(float('Nan')), data_dict['std_minutes_fair_active'].append(float('Nan'))
        data_dict['mean_minutes_vig_active'].append(float('Nan')), data_dict['std_minutes_vig_active'].append(float('Nan')),
        data_dict['mean_activity_calories'].append(float('Nan')), data_dict['std_activity_calories'].append(float('Nan'))
        data_dict['mean_minutes_sleep'].append(float('Nan')) , data_dict['std_minutes_sleep'].append(float('Nan'))
        data_dict['mean_minutes_rem'].append(float('Nan')), data_dict['std_minutes_rem'].append(float('Nan')),
        data_dict['mean_minutes_light'].append(float('Nan')), data_dict['std_minutes_light'].append(float('Nan'))
        data_dict['mean_minutes_deep'].append(float('Nan')), data_dict['std_minutes_deep'].append(float('Nan')),
        data_dict['mean_minutes_inbed'].append(float('Nan')), data_dict['std_minutes_inbed'].append(float('Nan'))
        data_dict['mean_number_awakenings'].append(float('Nan')), data_dict['std_number_awakenings'].append(float('Nan')) 
        


    #Adding the A1C from metabolism file
    if participant in metabolism[:,0]:
        #this part should be checked
        #indx = metabolism[metabolism[:,0]==participant,:]
        data_dict['a1c'].append(metabolism[metabolism[:,0]==participant][0][COL_A1C])
        data_dict['wc'].append(metabolism[metabolism[:,0]==participant, COL_WAIST])
        data_dict['bmi'].append(metabolism[metabolism[:,0]==participant, COL_BMI])
        data_dict['fbg'].append(metabolism[metabolism[:,0]==participant][0][COL_FBG])
        data_dict['tg'].append(metabolism[metabolism[:,0]==participant, COL_TG])
        data_dict['tchol'].append(metabolism[metabolism[:,0]==participant, COL_TCHol])
        data_dict['insulin'].append(metabolism[metabolism[:,0]==participant, COL_INSULIN])
    else:
        data_dict['a1c'].append(float('Nan'))
        data_dict['wc'].append(float('Nan'))
        data_dict['bmi'].append(float('Nan'))
        data_dict['fbg'].append(float('Nan'))
        data_dict['tg'].append(float('Nan'))
        data_dict['tchol'].append(float('Nan'))
        data_dict['insulin'].append(float('Nan'))
        

    #Adding the biomarkers from actigraph
    data_dict['weight'].append(float(actigraph[0,COL_WEIGHT]))
    data_dict['age'].append(float(actigraph[0,COL_AGE]))
    data_dict['gender'].append(actigraph[0,COL_GENDER])


dataframe = pd.DataFrame.from_dict(data_dict)
#dataframe.describe()

corrMatrix = dataframe.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()

#visualize the location of missing values
#sn.heatmap(dataframe.isnull(), cbar=False)
actigraph_variables = ['mean_steps', 'mean_morning_steps','mean_afternoon_steps', 'mean_evening_steps',
                        'std_steps',  'std_morning_steps',  'std_afternoon_steps',  'std_evening_steps',
                        'mean_acti_sed','std_acti_sed','mean_acti_light_active','std_acti_light_active',
                        'mean_acti_mv_active','std_acti_mv_active','mean_acti_vig_active','std_acti_vig_active']
fitbit_activity_variables = ['mean_calories_burnt', 'std_calories_burnt', 'mean_minutes_sed','std_minutes_sed',
                             'mean_minutes_light_active','std_minutes_light_active','mean_minutes_fair_active',
                             'std_minutes_fair_active','mean_minutes_vig_active','std_minutes_vig_active',
                             'mean_activity_calories','std_activity_calories']
metabolic_variables = ['gender', 'age','weight', 'wc', 'bmi','a1c','fbg']

varnames = ['gender', 'age','weight', 'wc', 'bmi', 'mean_steps', 'std_steps', 'mean_morning_steps', 
            'mean_afternoon_steps', 'mean_evening_steps', 
            'mean_calories_burnt', 'std_calories_burnt', 'mean_minutes_rem','std_minutes_rem']
#            'mean_minutes_rem', 'std_minutes_rem']
dep_varname = 'a1c'
Y = dataframe[dep_varname]
def without_keys(d, keys): 
    return {x: d[x] for x in d if x in keys}
X = without_keys(dataframe, varnames)
X = pd.DataFrame.from_dict(X)
#Adding gender as dummy variable
X_gender = pd.get_dummies(X['gender'])
X_new = X.drop(labels='gender', axis=1)
X_new = pd.concat([X_new, X_gender['M']], axis=1)
yf = Y.astype(float)

#X = dataframe.loc[ : , dataframe.columns.tolist() == varnames]
k = len(varnames)
RSS_list, R_squared_list, feature_list = [], [], []
numb_features = []
coeff_list = []


# Create a minimum and maximum processor object
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
# Create an object to transform the data to fit minmax processor
X_new_scaled = min_max_scaler.fit_transform(X_new)
#Looping over k = 1 to k = n features in X
est = sm.OLS(endog=yf, exog=X_new_scaled.astype(float), missing='drop')
est2 = est.fit()
#print(est2.ssr)
#print(est2.rsquared)
print(est2.summary())

from sklearn import linear_model

ynew = yf.values[~np.isnan(yf.values)]
X_hat = X_new.values.astype(float)[~np.isnan(yf.values),:]
clf = linear_model.Lasso(alpha=0.1, normalize = True)
#clf.fit(X_hat, ynew)


#PREPROCESSING ------------------------------------------------------------------------
dataframe_withA1C = dataframe
for i in range(110,121):
    dataframe_withA1C = dataframe_withA1C.drop(i,0)
    
dataframe['tchol'] = dataframe['tchol'].astype(float)
dataframe['tg'] = dataframe['tg'].astype(float)
dataframe['insulin'] = dataframe['insulin'].astype(float)
dataframe['wc'] = dataframe['wc'].astype(float)
dataframe['bmi'] = dataframe['bmi'].astype(float)


def without_keys(d, keys): 
    return {x: d[x] for x in d if x in keys}
X = without_keys(dataframe, varnames)
X = pd.DataFrame.from_dict(X) 
#Adding gender as dummy variable
X_gender = pd.get_dummies(X['gender'])
X_new = X.drop(labels='gender', axis=1)
X_new = pd.concat([X_new, X_gender['M']], axis=1)
#X_new = X_new.drop(labels='mean_minutes_rem', axis=1)
#X_new = X_new.drop(labels='std_minutes_rem', axis=1)

dep_varname = 'tchol'
y = dataframe[dep_varname]
y_new = y.astype(float)

#for i in range(71,110):
#    if (i != 93):
#        X_new = X_new.drop(i,0)
#        y_new = y_new.drop(i,0)

#Feature Selection
#X_feat = dataframe.to_numpy() #Use only the matrix data without headers
#X_feat = X_feat[:,1:-1] #Without Participant's ID
    
y = dataframe["a1c"] #Dependent variable
X = dataframe.drop("a1c",1) #Features only

#Using Pearson Correlation --------------------
plt.figure()
cor = dataframe.corr()
sn.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

corr_threshold = 0.5
dependentVar = abs(cor["mean_morning_steps"])
relevant_features = dependentVar[dependentVar>corr_threshold]
relevant_features
  #Check that they're not correlated among them (for regression)
  ##print(df[["VAR1","VAR2"]].corr())
#Feature Selection ------------------------------
  
#Iterative
#Backward elimination
#Removing categorical columns and dependent var
#dataframe_withA1C = dataframe
#for i in range(110,121):
#    dataframe_withA1C = dataframe_withA1C.drop(i,0)

y = dataframe_withA1C["a1c"] #Dependent variable
    
Xbe = dataframe_withA1C.drop("a1c",1)
Xbe = Xbe.drop("participant",1)
Xbe = Xbe.drop("gender",1)
X_gender = pd.get_dummies(dataframe_withA1C['gender'])
X_new = pd.concat([Xbe, X_gender['M']], axis=1)
Xbe_woNan = Xbe[['mean_morning_steps', 'mean_evening_steps','mean_afternoon_steps','std_morning_steps','std_evening_steps','std_afternoon_steps', 'wc','bmi','weight','age']]

#Xbe_woNan = X_new
#X_new = sm.add_constant(Xbe_woNan)
#Fitting sm.OLS model
min_max_scaler = preprocessing.MinMaxScaler()
# Create an object to transform the data to fit minmax processor
X_new_scaled = min_max_scaler.fit_transform(Xbe_woNan)
model = sm.OLS(y,X_new_scaled).fit()
model.summary()


cols = list(Xbe_woNan.columns)
pmax = 1
while (len(cols)>0):
    p = []
    X_1 = Xbe_woNan[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)




#RECURSIVE FEATURE ELIMINATION
actigraph_variables = [ 'mean_morning_steps','mean_afternoon_steps', 'mean_evening_steps',
                         'std_morning_steps','std_afternoon_steps', 'std_evening_steps',
                       'mean_acti_light_active',
                        'mean_acti_mv_active','mean_acti_vig_active',
                       # 'std_acti_sed','std_acti_light_active',
                       # 'std_acti_mv_active','std_acti_vig_active',
                        'gender', 'age', 'bmi']

#dataframe_new = dataframe_withA1C[(dataframe_withA1C['a1c']<11) & (dataframe_withA1C['a1c']>5)]
dataframe_new = dataframe_withA1C[(dataframe_withA1C['fbg']>=70) & (dataframe_withA1C['fbg']<170)]
#dataframe_new = dataframe_withA1C
y = dataframe_new["fbg"] #Dependent variable
X = without_keys(dataframe_new, actigraph_variables)
X = pd.DataFrame.from_dict(X) 
#X = pd.DataFrame(dict(X), index=[0])
#y= pd.DataFrame(dict(y), index=[0])
X = X.drop("gender",1)
X_gender = pd.get_dummies(dataframe_new['gender'])
X = pd.concat([X, X_gender['M']], axis=1)
min_max_scaler = preprocessing.MinMaxScaler()
X_new_scaled = min_max_scaler.fit_transform(X)

#Initializing RFE model
numberOfFeatures  = 6
rfe= RFE(model, numberOfFeatures)
#Transforming data using RFE
X_rfe = rfe.fit_transform(X,y)  
#Fitting the data to model
model.fit(X_rfe,y)
print(rfe.support_)
print(rfe.ranking_)


#Decision Tree ------------------------------

clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, y)
tree.plot_tree(clf) 
# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=3)
regr_2 = DecisionTreeRegressor(max_depth=4)
regr_1.fit(X, y)
regr_2.fit(X, y)

#Random Forest ------------------------------
#Making classes for the a1c
y = np.array(y)
y_class = y
for i in range(0, len(y)):
    if y[i] <= 100:#200:
        y_class[i] = 0
    elif y[i] < 130: # 240:
        y_class[i] = 1
    elif y[i] >= 130: #240:
        y_class[i] = 2

#y = y_class
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size = 0.2, random_state = 42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#y_test = sc.fit_transform(y)
#y_test = sc.fit_transform(np.array(y))
#rf=RandomForestClassifier(n_estimators=100)
#rf = RandomForestClassifier(max_depth=5, random_state=0)
# Train the model on training data
rf.fit(train_features, train_labels);
# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
#print("Accuracy:",metrics.accuracy_score(test_labels, predictions))
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
# Pull out one tree from the forest
tree = rf.estimators_[5]
# Export the image to a dot file
feature_list = X.columns
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
#graph.write_png('tree.png')
graph.write_png('tree.png', prog=['dot'])
# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(X, round(importance, 2)) for X, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
plt.plot(predictions,label = "lPredicted FBG")
plt.plot(np.array(test_labels),label = "Actual FBG")
plt.ylabel('Fasting Blood Glucose')
plt.title('Predicted Values Vs. Ground Truth')
plt.legend()
plt.show()

lenght = range(len(test_labels))
plt.scatter(lenght,test_labels)
plt.scatter(lenght,predictions)
plt.ylabel('Fasting Blood Glucose')
plt.title('Predicted Values Vs. Ground Truth')
plt.legend(["Ground Truth" , "Predicted Values"])
plt.legend()
plt.show()

 # Plot the actual values
plt.plot(np.array(test_labels), 'b-', label = 'actual')
# Plot the predicted values
plt.plot(np.array(predictions), 'ro', label = 'prediction')
plt.xticks(rotation = '60'); 
plt.legend()
# Graph labels
plt.xlabel('Participant'); plt.ylabel('Fasting Blood Glucose'); plt.title('Actual and Predicted Values');

df = pd.DataFrame(feature_importances, columns=['Features', 'Importance'])
df.plot(kind='barh', x='Features')


# New random forest with only the two most important variables
rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42)
# Extract the two most important features
train_important = without_keys(train_features, ['bmi','mean_morning_steps'])
train_important = pd.DataFrame.from_dict(train_important) 
test_important = without_keys(test_features, ['bmi','mean_morning_steps'])
test_important = pd.DataFrame.from_dict(test_important) 
# Train the random forest
rf_most_important.fit(train_important, train_labels)
# Make predictions and determine the error
predictions = rf_most_important.predict(test_important)
errors = abs(predictions - test_labels)
# Display the performance metrics
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
mape = np.mean(100 * (errors / test_labels))
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')

#------------------------------------------------------------

#finding the optimal number of features
X = Xbe_woNan
#no of features
nof_list=np.arange(1,13)            
high_score=0
#Variable to store the optimum features
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))


        

#Multiple Regression OLS ------------------------------\
actigraph_variables = [ 'mean_morning_steps','mean_afternoon_steps', 'mean_evening_steps',
                        # 'std_morning_steps','std_afternoon_steps', 'std_evening_steps',
                        'mean_acti_light_active', #'mean_acti_sed',
                       # 'mean_acti_mv_active',#'mean_acti_vig_active',
                      #  'std_acti_sed','std_acti_light_active',
                      #  'std_acti_mv_active','std_acti_vig_active',
                        'bmi']

#dataframe_new = dataframe_withA1C[(dataframe_withA1C['a1c']<11) & (dataframe_withA1C['a1c']>5)]
dataframe_new = dataframe_withA1C[(dataframe_withA1C['age']>=60) ]
#dataframe_new = dataframe_withA1C
y = dataframe_new["fbg"] #Dependent variable
X = without_keys(dataframe_new, actigraph_variables)
X = pd.DataFrame.from_dict(X) 
#X = pd.DataFrame(dict(X), index=[0])
#y= pd.DataFrame(dict(y), index=[0])
X = X.drop("gender",1)
X_gender = pd.get_dummies(dataframe_new['gender'])
X = pd.concat([X, X_gender['M']], axis=1)
min_max_scaler = preprocessing.MinMaxScaler()
X_new_scaled = min_max_scaler.fit_transform(X)

actigraph_variables = ['mean_steps', 'mean_morning_steps','mean_afternoon_steps', 'mean_evening_steps',
                        'std_steps',  'std_morning_steps',  'std_afternoon_steps',  'std_evening_steps',
                        'mean_acti_sed','std_acti_sed','mean_acti_light_active','std_acti_light_active',
                        'mean_acti_mv_active','std_acti_mv_active','mean_acti_vig_active','std_acti_vig_active',
                        'gender', 'age', 'wc','weight','bmi']
both_variables = [ 'mean_morning_steps','mean_afternoon_steps', 'mean_evening_steps',
                          'std_morning_steps',  'std_afternoon_steps',  'std_evening_steps',
                        'mean_acti_vig_active','std_acti_vig_active',
                        'gender', 'age', 'wc','weight','bmi','mean_calories_burnt', 'std_calories_burnt', 'mean_minutes_sed','std_minutes_sed',
                             'mean_minutes_light_active','std_minutes_light_active','mean_minutes_vig_active','std_minutes_vig_active',
                             'mean_activity_calories','std_activity_calories']
actigraph_variables = [ 'mean_morning_steps','mean_afternoon_steps', 'mean_evening_steps',
                          'std_morning_steps',  'std_afternoon_steps',  'std_evening_steps',
                        'mean_acti_vig_active','std_acti_vig_active',
                        'gender', 'age', 'wc','weight','bmi']
dataframe_new = dataframe_withA1C[(dataframe_withA1C['a1c']<13) & (dataframe_withA1C['a1c']>5)]
y = dataframe_new["a1c"] #Dependent variable
X = without_keys(dataframe_new, actigraph_variables)
X = pd.DataFrame.from_dict(X) 
#X = pd.DataFrame(dict(X), index=[0])
#y= pd.DataFrame(dict(y), index=[0])
X = X.drop("gender",1)
X_gender = pd.get_dummies(dataframe_new['gender'])
X = pd.concat([X, X_gender['M']], axis=1)
#X = sm.add_constant(X)
min_max_scaler = preprocessing.MinMaxScaler()
X_new_scaled = min_max_scaler.fit_transform(X)
#Looping over k = 1 to k = n features in X
est = sm.OLS(endog=y, exog=X.astype(float), missing='drop')
est2 = est.fit()
#print(est2.ssr)
#print(est2.rsquared)
print(est2.summary())
ynewpred =  est2.predict(X) # predict out of sample
#print(ynewpred)
plt.plot(np.array(ynewpred),label = "lPredicted FBG")
plt.plot(np.array(y),label = "Actual FBG")
plt.ylabel('Fasting Blood Glucose')
plt.title('Predicted Values Vs. Ground Truth')
plt.legend()
plt.show()

#Feature Selection LASSO ------------------------------

reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X_new_scaled,y))
coef = pd.Series(reg.coef_, index = X.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")



#Feature Selection LASSO ------------------------------
#EMBEDDED
def isnt_numerical(type_tuple: Tuple[str, dtype]) -> bool:
    """
    Checks whether the input tuple of form
    (<column name>, <column type>) is describing a numerical column
    :param type_tuple: a type of form ("column name", dtype('column type'))
    :return: A boolean indicating whether the column is numerical
    """
    return type_tuple[1] != dtype('float64') and type_tuple[1] != dtype('int64')

def divide_columns_type(
        data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Get columns of a Pandas DataFrame that are numerical or otherwise not excluded
    from the DataFrame
    :param data: The DataFrame to extract columns from
    :return: A tuple of the numerical subset of the DataFrame and the total
        list of columns excluded from the DataFrame
    """
    new_columns_excluded = [
        tup[0] for tup in list(
            filter(isnt_numerical,
                   data.reset_index().dtypes.to_dict().items()))
    ]

    columns_included = [
        item for item in data.columns.to_list()
        if item not in new_columns_excluded
    ]

    inc_data = data.loc[:, columns_included]

    return inc_data, columns_included, new_columns_excluded

def show_density(data: pd.DataFrame) -> None:
    """
    Plot each numerical column of the input data in a histogram,
    useful for visualizing data distribution.
    :param data: a Pandas DataFrame with at least 1 numerical column
    :return: Does not return, draws a histogram instead.
    """

    num_data, columns_included, _ = divide_columns_type(data)

    square_size = int(math.ceil(math.sqrt(len(num_data.columns.to_list()))))

    fig, axes = plt.subplots(square_size, square_size, squeeze=False)

    fig.suptitle('Density distributions for each column of the data')
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    for row in range(0, square_size):
        for col in range(0, square_size):
            ind = row * square_size + col
            if ind < len(columns_included):
                axes[row, col].hist(num_data[columns_included[ind]],
                                    histtype='step',
                                    bins=40,
                                    density=True)
                axes[row, col].title.set_text(columns_included[ind])
                #axes[row, col].set_xlabel("Value")
                #axes[row, col].set_ylabel("Frequency of Value (Normalized)")

    plt.show()