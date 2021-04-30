# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 11:20:49 2020

@author: aijing
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.markers as mk
import seaborn as sns
import numpy as np
import scipy as sp
from sklearn.preprocessing import MinMaxScaler
import ipdb
import warnings
warnings.filterwarnings("ignore")


img_07_2019=pd.read_csv('data_07 downsample.csv').drop(['row','col','longitude','latitude','GRVI', 'EXG', 'EXG_EXR','H','canopy_size_raw','Lab_a'],axis=1)
img_08_2019=pd.read_csv('data_08 downsample.csv').drop(['row','col','longitude','latitude','GRVI', 'EXG', 'EXG_EXR','H','Lab_a','canopy_size_raw'],axis=1)
img_09_2019=pd.read_csv('data_09 downsample.csv').drop(['row','col','longitude','latitude','GRVI', 'EXG', 'EXG_EXR','H','Lab_a','canopy_size_raw'],axis=1)
img_09_2018=pd.read_csv('data_09_2018 downsample.csv').drop(['row','col','longitude','latitude','easting_16N','northing_16N'],axis=1)
img_08_2018=pd.read_csv('data_08_2018 downsample.csv').drop(['row','col','longitude','latitude','easting_16N','northing_16N','thermal','x','y'],axis=1)
img_07_2018=pd.read_csv('data_07_2018 downsample.csv').drop(['row','col','longitude','latitude','easting_16N','northing_16N','NDVI_raw','canopy_size_raw','GNDVI_raw','NDRE'],axis=1)
img_06_2018=pd.read_csv('data_06_2018 downsample.csv').drop(['row','col','longitude','latitude','easting_16N','northing_16N','NDVI_raw','canopy_size_raw','GNDVI_raw','NDRE'],axis=1)
img_08_2017=pd.read_csv('data_08_2017 downsample.csv').drop(['row','col','longitude','latitude','thermal','NDVI_raw','canopy_size_raw','GNDVI_raw','NDRE'],axis=1)

'''
soil_W=pd.read_csv('soil features downsample W.csv').drop(['row','col','longitude','latitude','irrigation_length','irrigation_circle','sand%','clay%',
                'WP','FC','TAW','RAW','Saturation_cm3water/cm3soil','Drainage rate_cm/hr','Drainage rate_cm/hr','clay10','clay20','clay30','clay40','clay50','clay60','clay70'],axis=1)
soil_E=pd.read_csv('soil features downsample.csv').drop(['row','col','longitude','latitude','irrigation_length','irrigation_circle','sand%','clay%',
                'WP','FC','TAW','RAW','Saturation_cm3water/cm3soil','Drainage rate_cm/hr','Drainage rate_cm/hr','clay10','clay20','clay30','clay40','clay50','clay60','clay70'],axis=1)
soil_E_120=pd.read_csv('soil features downsample - 09.csv').drop(['row','col','longitude','latitude','irrigation_length','irrigation_circle','sand%','clay%',
                'WP','FC','TAW','RAW','Saturation_cm3water/cm3soil','Drainage rate_cm/hr','Drainage rate_cm/hr','clay10','clay20','clay30','clay40','clay50','clay60','clay70'],axis=1)


'''

soil_W=pd.read_csv('soil features downsample W.csv')[['sand%','WP','FC','TAW',
                'clay10','clay20','clay30','clay40','clay50','clay60','clay70']]
soil_E=pd.read_csv('soil features downsample.csv')[['sand%','WP','FC','TAW',
                'clay10','clay20','clay30','clay40','clay50','clay60','clay70']]
soil_E_120=pd.read_csv('soil features downsample - 09.csv')[['sand%','WP','FC','TAW',
                'clay10','clay20','clay30','clay40','clay50','clay60','clay70']]


row_col_W=pd.read_csv('soil features downsample W.csv')[['row','col']]
row_col_E=pd.read_csv('soil features downsample.csv')[['row','col']]
row_col_E_120=pd.read_csv('soil features downsample - 09.csv')[['row','col']]


irrigation_E=pd.read_csv('soil features downsample.csv')[['irrigation_length', 'irrigation_circle']]
irrigation_E_120=pd.read_csv('soil features downsample - 09.csv')[['irrigation_length', 'irrigation_circle']]
irrigation_W=pd.read_csv('soil features downsample W.csv')[['irrigation_length', 'irrigation_circle']]

irrigation_data_2019=pd.read_csv('irrigation data 2019.csv')
irrigation_data_2018=pd.read_csv('irrigation data 2018.csv')


weather_2017=pd.read_csv('weather data 2017.csv')[['acc rain', 'acc GDD','acc ET']]
weather_2018=pd.read_csv('weather data 2018.csv')[['acc rain', 'acc GDD','acc ET']]
weather_2019=pd.read_csv('weather data 2019.csv')[['acc rain', 'acc GDD','acc ET']]

col_names=['I_ks','T_<1','L_<1','T_0.9_1','L_<0.9','T_0.8_0.9','L_<0.8','T_0.7_0.8',
           'L_0.7','T_0.6_0.7','L_<0.6','T_0.5_0.6','L_<0.5','T_0.4_0.5','L_<0.4','T<0.4',
           'L_<0.3','B<1']
ks_07_2019=pd.read_csv('ks features 072019.csv',names=col_names,header=0)
ks_08_2019=pd.read_csv('ks features 082019.csv',names=col_names,header=0)
ks_09_2019=pd.read_csv('ks features 092019.csv',names=col_names,header=0)
ks_06_2018=pd.read_csv('ks features 062018.csv',names=col_names,header=0)
ks_07_2018=pd.read_csv('ks features 072018.csv',names=col_names,header=0)
ks_08_2018=pd.read_csv('ks features 082018.csv',names=col_names,header=0)
ks_09_2018=pd.read_csv('ks features 092018.csv',names=col_names,header=0)
ks_08_2017=pd.read_csv('ks features 082017.csv',names=col_names,header=0)

ks_2017_H=pd.read_csv('ks features 2017 harvest.csv',names=col_names,header=0)
ks_2018_H=pd.read_csv('ks features 2018 harvest.csv',names=col_names,header=0)
ks_2019_H=pd.read_csv('ks features 2019 harvest.csv',names=col_names,header=0)
ks_2019_H_09=pd.read_csv('ks features 2019 09 harvest.csv',names=col_names,header=0)


DAP={
  "201907": 57,
  "201908": 90,
  "201909": 113,
  "2019H": 158,
  "201806": 44,
  "201807": 63,
  "201808": 98,
  "201809": 122,
  "2018H": 148,
  "201708": 81,
  "2017H": 135
}

flag=pd.DataFrame([['2019','09'],['2019','08'],['2019','07'],
                   ['2018','09'],['2018','08'],['2018','07'],['2018','06'],
                   ['2017','08']])
data_all={}
for i in ('img_07_2019', 'img_08_2019', 'img_09_2019','img_09_2018','img_08_2018',
          'img_07_2018','img_06_2018','img_08_2017','soil_W','soil_E','soil_E_120',
          'irrigation_data_2019','irrigation_data_2018','weather_2017',
          'weather_2018','weather_2019','irrigation_E','irrigation_E_120','irrigation_W',
          'row_col_W', 'row_col_E', 'row_col_E_120','ks_07_2019','ks_08_2019','ks_09_2019',
          'ks_06_2018','ks_07_2018','ks_08_2018','ks_09_2018','ks_08_2017',
          'ks_2017_H', 'ks_2018_H','ks_2019_H', 'ks_2019_H_09'):
    data_all[i] = locals()[i]
#--------------------  Pearson correlation  ---------------------------------------------
'''
month='06'
year='2018'
index=np.where((data_all['img_'+month+'_'+year]['NDVI']>0.6) &
                   (data_all['img_'+month+'_'+year]['Yield'+year]>100) &
                    (data_all['img_'+month+'_'+year]['GNDVI']>0.6) &
                    (data_all['img_'+month+'_'+year]['canopy_size']>-0.01) &
                    (data_all['img_'+month+'_'+year]['a'].notna()) &
                    (data_all['img_'+month+'_'+year]['thermal'].notna()))
x=pd.concat([data_all['img_'+month+'_'+year].iloc[index[0],:],soil_E_120.iloc[index[0],:],ks_09_2019.iloc[index[0],:]],axis=1)
x=pd.concat([data_all['img_'+month+'_'+year].iloc[index[0],:],soil_W.iloc[index[0],:]],axis=1)
y=data_all['img_'+month+'_'+year].iloc[index[0],4]
#x=pd.concat([data_all['img_07_2018'].iloc[:,0:3],data_all['img_06_2018'].iloc[:,0:4]],axis=1)

x=pd.concat([data_all['img_08_2017'],data_all['img_09_2019']],axis=1)

x=pd.concat([soil_E.iloc[:,-23:-8],soil_W.iloc[:,-23:-8]],axis=0)
corr=round(x.corr(method='pearson'),2)
colormap = plt.cm.RdBu
colormap = plt.cm.Reds
mask = np.zeros_like(x.astype(float).corr())
mask[np.triu_indices_from(mask)] = True
sns.heatmap(x.astype(float).corr().round(2),linewidths=0.1,vmax=0.9,vmin=-0.9, 
            square=True, cmap=colormap, linecolor='white', annot=True,mask=mask, annot_kws={"size": 10})

x=pd.DataFrame(min_max_scaler.fit_transform(x), columns=x.columns)

'''
#-----------------------Saxton Equations ----------------------------------------------
soil_W=pd.read_csv('soil features downsample W.csv')[['sand%','clay%','WP','FC','TAW',
                'clay10','clay20','clay30','clay40','clay50','clay60','clay70',
                'clayD1','clayD2','clayD3','clayD4','clayD5','clayD6','clayD7','clayD8',
                'sandD1','sandD2','sandD3','sandD4','sandD5','sandD6','sandD7','sandD8']]
soil_E=pd.read_csv('soil features downsample.csv')[['sand%','clay%','WP','FC','TAW',
                'clay10','clay20','clay30','clay40','clay50','clay60','clay70',
                'clayD1','clayD2','clayD3','clayD4','clayD5','clayD6','clayD7','clayD8',
                'sandD1','sandD2','sandD3','sandD4','sandD5','sandD6','sandD7','sandD8']]
soil_E_120=pd.read_csv('soil features downsample - 09.csv')[['sand%','clay%','WP','FC','TAW',
                'clay10','clay20','clay30','clay40','clay50','clay60','clay70',
                'clayD1','clayD2','clayD3','clayD4','clayD5','clayD6','clayD7','clayD8',
                'sandD1','sandD2','sandD3','sandD4','sandD5','sandD6','sandD7','sandD8']]


clay=soil_E_120.iloc[:,-16:-8]
sand=soil_E_120.iloc[:,-8:]

WP=np.zeros((len(clay),8))
FC=np.zeros((len(clay),8))

for i in range(8):
    clay_temp=clay.iloc[:,i]
    sand_temp=sand.iloc[:,i]

    a=np.exp(-4.396-0.0715*clay_temp-0.000488*sand_temp*sand_temp-0.00004285*sand_temp*sand_temp*clay_temp)
    b=-3.14-0.00222*clay_temp*clay_temp-0.00003484*sand_temp*sand_temp*clay_temp

    WP[:,i]=np.power((15/a),(1/b))
    FC[:,i]=np.power((0.333333/a),(1/b))
    
#-------------------------- soil FC and WP plot ---------------------------------------
soil_W=pd.read_csv('soil features downsample W.csv')[[
                'WP_D1','WP_D2','WP_D3','WP_D4','WP_D5','WP_D6','WP_D7','WP_D8',
                'FC_D1','FC_D2','FC_D3','FC_D4','FC_D5','FC_D6','FC_D7','FC_D8']]
soil_E=pd.read_csv('soil features downsample.csv')[[
                'WP_D1','WP_D2','WP_D3','WP_D4','WP_D5','WP_D6','WP_D7','WP_D8',
                'FC_D1','FC_D2','FC_D3','FC_D4','FC_D5','FC_D6','FC_D7','FC_D8']]

WP_FC=soil_E
for i in range(16):
    WP_FC_temp=WP_FC.iloc[:,i]
    WP_FC_temp=np.resize(WP_FC_temp,(63,38))
    
    if i>7:
        vmax_value=0.35
    else:
        vmax_value=0.2
    plt.imshow(WP_FC_temp,cmap='jet',vmin=0, vmax=vmax_value)
    plt.axis('off')
    plt.colorbar()

    plt.savefig('soil_E'+str(i)+'.png', dpi=150)
    plt.close()
    
#----------------------   Data combining  ---------------------------------------------
# for yield prediction
data_pd=pd.DataFrame([])
row_col_pd=pd.DataFrame([])
Yield=[]
for _,[year, month] in flag.iterrows():
    #print(year,month)
    date_data=pd.DataFrame(np.repeat(np.resize([year,month],(1,2)),len(data_all['img_'+month+'_'+year]),axis=0),columns=['year','month'])
    DAP_data=pd.DataFrame(np.repeat(np.resize([DAP[year+month],DAP[year+'H']],(1,2)),len(data_all['img_'+month+'_'+year]),axis=0),columns=['Imaging_DAP','Harvest_DAP'])
    
    weather_temp=np.repeat(np.resize([data_all['weather_'+year].loc[DAP[year+month]],
                                      data_all['weather_'+year].loc[DAP[year+'H']]],(1,6)),
                                      len(data_all['img_'+month+'_'+year]),axis=0)
    
    if year=='2017':
        soil='soil_E'
        rolcol='row_col_E'
        data_combine_flag=1;
    else:
        soil='soil_E'
        rolcol='row_col_E'
        data_combine_flag=0;
    if year=='2018':
        soil='soil_W'
        rolcol='row_col_W'
        irrigation='irrigation_W'
    if (year=='2019') and (month=='07' or month=='08') :
        soil='soil_E'
        rolcol='row_col_E'
        irrigation='irrigation_E'
    if (year=='2019') and (month=='09') :
        soil='soil_E_120'
        rolcol='row_col_E_120'
        irrigation='irrigation_E_120'
        
    if data_combine_flag==0:    
        for i,I in enumerate(data_all['irrigation_data_'+year].columns):
            if i>1:
               if DAP[year+month]>int(I):
                   #print(I)
                   index=np.where(data_all['irrigation_data_'+year][I]!=0)
                   irrigation_plot=np.array(data_all['irrigation_data_'+year][['irrigation_length','irrigation_circle']])[index[0]]
                   for irrigation_length, irrigation_circle in irrigation_plot:
                       #print(irrigation_length, irrigation_circle)
                       weather_temp[np.ix_((data_all[irrigation].irrigation_length==irrigation_length) & 
                                    (data_all[irrigation].irrigation_circle==irrigation_circle),[0,3])]+=float(data_all['irrigation_data_'+year][(data_all['irrigation_data_'+year].irrigation_length==irrigation_length)
                                    & (data_all['irrigation_data_'+year].irrigation_circle==irrigation_circle)][I])
            
               else:
                   index=np.where(data_all['irrigation_data_'+year][I]!=0)
                   irrigation_plot=np.array(data_all['irrigation_data_'+year][['irrigation_length','irrigation_circle']])[index[0]]
                   for irrigation_length, irrigation_circle in irrigation_plot:
                       #print(irrigation_length, irrigation_circle)
                       weather_temp[(data_all[irrigation].irrigation_length==irrigation_length) & 
                                    (data_all[irrigation].irrigation_circle==irrigation_circle),3]+=float(data_all['irrigation_data_'+year][(data_all['irrigation_data_'+year].irrigation_length==irrigation_length)
                                    & (data_all['irrigation_data_'+year].irrigation_circle==irrigation_circle)][I])
       
    
    weather_data=pd.DataFrame(weather_temp, columns=['rain_imaging','GDD_imaging','ET_imaging',
                                                     'rain_harvest','GDD_harvest','ET_harvest'])          
    
    index=np.where((data_all['img_'+month+'_'+year]['NDVI']>0.6) &
                   (data_all['img_'+month+'_'+year]['Yield'+year]>100) &
                    (data_all['img_'+month+'_'+year]['GNDVI']>0.6) &
                    (data_all['img_'+month+'_'+year]['canopy_size']>-0.01) &
                    (data_all['img_'+month+'_'+year]['a'].notna()))
    data_temp=pd.concat([data_all['img_'+month+'_'+year].iloc[index[0],0:4],
                            data_all[soil].iloc[index[0],:],DAP_data.iloc[index[0],:],
                            weather_data.iloc[index[0],:]],axis=1)
    Yield=np.append(Yield,data_all['img_'+month+'_'+year].iloc[index[0],4])
    data_pd=data_pd.append(data_temp)
    row_col_pd=row_col_pd.append(pd.concat([date_data.iloc[index[0],:],data_all[rolcol].iloc[index[0],:]], axis=1))
    
data_pd=data_pd.reset_index(drop=True)
row_col_pd=row_col_pd.reset_index(drop=True)


#--------------------------only for testing -------------------------------------------------------------        
# data set of soil to predict image features
x=pd.concat([soil_W,pd.DataFrame(img_08_2018['a'])],axis=1)
x=x.dropna()
y=x['a']
x=x.drop(['a'],axis=1)
#x=pd.DataFrame(min_max_scaler.fit_transform(x), columns=x.columns)

x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.15, random_state=39)

# data set of ks to predict image features
x=pd.concat([ks_07_2018,pd.DataFrame(img_07_2018['NDVI'])],axis=1)
x=x.dropna()

import re
regex = re.compile(r"\[|\]|<", re.IGNORECASE)
x.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in x.columns.values]


y=x['NDVI']
x=x.drop(['NDVI'],axis=1)
#x=pd.DataFrame(min_max_scaler.fit_transform(x), columns=x.columns)

x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.15, random_state=39)

# data set of ks+soil to predict image features
x=pd.concat([soil_W,ks_06_2018,pd.DataFrame(img_06_2018['NDVI'])],axis=1)
x=x.dropna()

import re
regex = re.compile(r"\[|\]|<", re.IGNORECASE)
x.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in x.columns.values]


y=x['NDVI']
x=x.drop(['NDVI'],axis=1)
x=pd.DataFrame(min_max_scaler.fit_transform(x), columns=x.columns)

x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.15, random_state=39)

# data set of ks+soil to predict five image features together
x=pd.concat([soil_W,ks_08_2018,img_08_2018.iloc[:,0:5]],axis=1)
x=x.dropna()

import re
regex = re.compile(r"\[|\]|<", re.IGNORECASE)
x.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in x.columns.values]


y=x.iloc[:,31:36]
x=x.drop(['NDVI','canopy_size','GNDVI','a','NDRE'],axis=1)
x=pd.DataFrame(min_max_scaler.fit_transform(x), columns=x.columns)

x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.15, random_state=39)


import pickle
# save the model to disk
filename = 'x_train.pkl'
#pickle.dump(y_test, open(filename, 'wb'))
x_train = pickle.load(open(filename, 'rb'))

#----------------------   Linear model   -----------------------------------------------
import statsmodels.api as sm
from statistics import mean
import math

x_train = sm.add_constant(x_train,has_constant='add')
x_test = sm.add_constant(x_test,has_constant='add')

model = sm.OLS(y_train,np.asanyarray(x_train))
model_results = model.fit()
predictions = model_results.predict(np.asanyarray(x_test))
model_results.summary()

# plot
z = np.polyfit(predictions, y_test, 1)
p = np.poly1d(z)

yhat = p(predictions)
ybar = sum(y_test)/len(y_test)
SST = sum((y_test - ybar)**2)
SSreg = sum((yhat - ybar)**2)

R2 = SSreg/SST

MAE = mean(abs(yhat-y_test))
MAPE= mean(abs(yhat-y_test)/y_test)
RMSE= math.sqrt(mean((yhat-y_test)**2))

#color =(pd.to_numeric(row_col_test['month'])*250+pd.to_numeric(row_col_test['year']))/4269
plt.scatter(x=predictions, y=y_test, cmap=None)
plt.plot(predictions,p(predictions),"r--")
plt.text(0.15, 0.95, 'y='+str(round(z[0],2))+'x+'+str(round(z[1],2)), size=12,horizontalalignment='center',verticalalignment='center',transform=plt.gca().transAxes)
plt.text(0.15, 0.9, 'R2='+str(round(R2,2)), size=12,horizontalalignment='center',verticalalignment='center',transform=plt.gca().transAxes)
plt.text(0.15, 0.85, 'MAE='+str(round(MAE,2)), size=12,horizontalalignment='center',verticalalignment='center',transform=plt.gca().transAxes)
plt.text(0.15, 0.8, 'MAPE='+str(round(MAPE*100,2))+'%', size=12,horizontalalignment='center',verticalalignment='center',transform=plt.gca().transAxes)
plt.text(0.15, 0.75, 'RMSE='+str(round(RMSE,2)), size=12,horizontalalignment='center',verticalalignment='center',transform=plt.gca().transAxes)
plt.show()

# single image feature to predict yield
slop=pd.DataFrame([])
for _,[year, month] in flag.iterrows():
    for image_features in data_all['img_'+month+'_'+year].columns:
        
        if image_features=='Yield'+year:
            continue
        x=pd.concat([data_all['img_'+month+'_'+year][image_features],data_all['img_'+month+'_'+year]['Yield'+year]],axis=1)
        x=x.dropna()
        x_train=x[image_features]
        y_train=x['Yield'+year]
        x_train = sm.add_constant(x_train,has_constant='add')
        
        model = sm.OLS(y_train,np.asanyarray(x_train))
        model_results = model.fit()
        predictions = model_results.predict(np.asanyarray(x_train))
        
        z = np.polyfit(predictions, y_train, 1)
        p = np.poly1d(z)
        
        yhat = p(predictions)
        ybar = sum(y_train)/len(y_train)
        SST = sum((y_train - ybar)**2)
        SSreg = sum((yhat - ybar)**2)
        R2 = SSreg/SST
        
        
        temp=pd.DataFrame(np.resize(np.array((round(model_results.params['x1'],2),round(R2,2))),(1,2)))
        temp.rename(index={0:image_features+'_'+month+'_'+year}, inplace=True)
        temp.rename(columns ={0:'slope',1:'r2'},inplace=True)
        slop=slop.append(temp)


        
        
#----------------------   XGBoost    -----------------------------------------------
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics 


x_train = x_train.drop(['const'],axis=1)
x_test = x_test.drop(['const'],axis=1)

tuned_parameters = {'n_estimators': [200,300,400,500,600],
                     'max_depth': [3,4,5,6,7],
                     'C': [0.01,0.1, 1],
                     'learning_rate': [0.05,0.1,0.15,0.2],
                     'reg_lambda':[0.2,0.4,0.5,0.6,0.8]}
                    
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror')
clf = GridSearchCV(xg_reg, tuned_parameters, scoring='neg_mean_absolute_error',cv=5)
clf.fit(x_train, y_train)

means = clf.cv_results_['mean_test_score']
clf.best_params_
model=clf.best_estimator_
predictions=model.predict(x_test)

# go to the plot in Linear model
xgb.plot_tree(model,num_trees=5)
xgb.plot_importance(model,importance_type='weight')  # gain or weight

# Plot the feature importances of the forest
importances =pd.DataFrame(np.resize(model.feature_importances_,
                                    (1,len(x_train.columns))),
                                    columns=x_train.columns)
indices = np.array(np.argsort(importances))
sort=importances.iloc[0,indices[0]]
plt.figure()
plt.title("Feature importances")
plt.barh(range(x_train.shape[1]), sort,color="r",  align="center")
# If you want to define your own labels,
# change sort.index to a list of labels on the following line.
plt.yticks(range(x_train.shape[1]), sort.index)
plt.ylim([-1, x_train.shape[1]])
plt.show()


import pickle
# save the model to disk
filename = 'XGBoost_model.pkl'
pickle.dump(model, open(filename, 'wb'))
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
predictions=model.predict(x_test)


# use BayesianOptimization for parameter search
# for image features prediction using soil+ks

import re
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

import xgboost as xgb
from sklearn.metrics import r2_score
from bayes_opt import BayesianOptimization
import statsmodels.api as sm
from statistics import mean
import math
import pickle
import matplotlib

global x_train, x_test, y_train, y_test
#r2=np.zeros((6, 8))
#r2_i=0
#r2_j=0
feature_importance=pd.DataFrame([])
def xgb_r2_score(preds, dtrain):
    # Courtesy of Tilii
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)

def train_xgb(n_trees, eta,max_depth, subsample,C, reg_lambda):
    # Evaluate an XGBoost model using given params
    xgb_params = {
        'n_trees': int(n_trees),
        'eta': eta,
        'max_depth': int(max_depth),
        'subsample': max(min(subsample, 1), 0),
        'objective': 'reg:squarederror',
        'silent': 1,  #0: print the processing result, 1: not print 
        'C': max(C, 0),
        'reg_lambda': max(reg_lambda,0)
    }
    #scores = xgb.cv(xgb_params, dtrain, num_boost_round=1500, early_stopping_rounds=50, verbose_eval=False, feval=xgb_r2_score, maximize=True, nfold=5)['test-r2-mean'].iloc[-1]
    
    model = xgb.XGBRegressor(**xgb_params,feval=xgb_r2_score, maximize=True)
    global x_train, x_test, y_train, y_test
    model.fit(x_train,y_train)    
    predictions = model.predict(x_test)
    
    z = np.polyfit(predictions, y_test, 1)
    p = np.poly1d(z)
    yhat = p(predictions)
    ybar = sum(y_test)/len(y_test)
    SST = sum((y_test - ybar)**2)
    SSreg = sum((yhat - ybar)**2)
        
    R2 = SSreg/SST
    return R2

params = {
  'n_trees':(20, 250),
  'eta': (0.001,2),
  'C':(10e-2, 1),
  'subsample':(0.1, 0.9),
  'max_depth': (3, 10),
  'reg_lambda':(0.1,0.9)
}

targets=pd.DataFrame([],columns=['2019_07', '2019_08', '2019_09',
                                 '2018_06','2018_07', '2018_08', '2018_09',
                                 '2017_08'], index=['NDVI','canopy_size','GNDVI','NDRE','a',
                                                    'Yield2019','Yield2018','Yield2017'])
font = {'size'   : 11}
matplotlib.rc('font', **font)         
                                         
for _,[year, month] in flag.iterrows():
    #print(year,month)
       
    if year=='2017':
        soil='soil_E'
        havest='ks_2017_H'
    if year=='2018':
        soil='soil_W'
        havest='ks_2018_H'
    if (year=='2019') and (month=='07' or month=='08') :
        soil='soil_E'
        havest='ks_2019_H'
    if (year=='2019') and (month=='09') :
        soil='soil_E_120'
        havest='ks_2019_H_09'
    
    #r2_i=0
    for image_features in data_all['img_'+month+'_'+year].columns:
        
        print(image_features)
        #ipdb.set_trace()
        '''
        if image_features=='Yield'+year:
            continue
        '''
        print(year,month,image_features)
        index=np.where((data_all['img_'+month+'_'+year]['NDVI']>0.6) &
                   (data_all['img_'+month+'_'+year]['Yield'+year]>100) &
                    (data_all['img_'+month+'_'+year]['GNDVI']>0.6))
        x=pd.concat([data_all[soil].iloc[index[0],:],data_all['ks_'+month+'_'+year].iloc[index[0],:],pd.DataFrame(data_all['img_'+month+'_'+year].iloc[index[0],:][image_features])],axis=1)
        x=x.dropna()
        y=x[image_features]
        x=x.drop([image_features],axis=1)
        
        '''
        index=np.where((data_all['img_'+month+'_'+year]['NDVI']>0.6) &
                   (data_all['img_'+month+'_'+year]['Yield'+year]>100) &
                    (data_all['img_'+month+'_'+year]['GNDVI']>0.6))
        x=pd.concat([data_all['img_'+month+'_'+year].iloc[index[0],0:3],data_all[soil].iloc[index[0],:],data_all[havest].iloc[index[0],:]],axis=1)
        y=data_all['img_'+month+'_'+year].iloc[index[0],-1]
        '''
        
        regex = re.compile(r"\[|\]|<", re.IGNORECASE)
        x.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in x.columns.values]
        '''
        x_train.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in x_train.columns.values]
        x_test.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in x_test.columns.values]
        '''
        
        x=pd.DataFrame(min_max_scaler.fit_transform(x), columns=x.columns)
        global x_train, x_test, y_train, y_test
        x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.15, random_state=39)
        
        '''
        dtrain = xgb.DMatrix(x_train, y_train)
        dtest = xgb.DMatrix(x_test)
        '''
        
        xgb_bayesopt = BayesianOptimization(train_xgb, params)

        # Maximize R2 score
        xgb_bayesopt.maximize(init_points=10, n_iter=200)
        
        # Get the best params
        p = xgb_bayesopt.max['params']
        target=xgb_bayesopt.max['target']
        
        
        xgb_params = {
            'n_trees': int(p['n_trees']),
            'eta': p['eta'],
            'max_depth': int(p['max_depth']),
            'subsample': max(min(p['subsample'], 1), 0),
            'objective': 'reg:squarederror',
            'silent': 1,
            'C': max(p['C'],0),
            'reg_lambda':max(p['reg_lambda'],0)
        }
        '''
        #model = xgb.train(xgb_params, dtrain, num_boost_round=1500, verbose_eval=False, feval=xgb_r2_score, maximize=True)
        f = open('./xgboost_results/xgb_params_'+image_features+month+'_'+year+'.pkl',"rb")
        xgb_params=pickle.load(f)
        f.close()
        '''
        model = xgb.XGBRegressor(**xgb_params,feval=xgb_r2_score, maximize=True)

        model.fit(x_train,y_train)
        
        predictions = model.predict(x_test)
        
        f = open('./xgboost_results/xgb_params_'+image_features+month+'_'+year+'.pkl',"wb")
        pickle.dump(xgb_params,f)
        f.close()
        
        # plot
        z = np.polyfit(predictions, y_test, 1)
        p = np.poly1d(z)
        
        yhat = p(predictions)
        ybar = sum(y_test)/len(y_test)
        SST = sum((y_test - ybar)**2)
        SSreg = sum((yhat - ybar)**2)
        
        R2 = SSreg/SST
        
        #r2[r2_i,r2_j]=R2
        #r2_i=r2_i+1
        
        MAE = mean(abs(yhat-y_test))
        MAPE= mean(abs(yhat-y_test)/y_test)
        RMSE= math.sqrt(mean((yhat-y_test)**2))
        
        
        plt.scatter(x=predictions, y=y_test, cmap=None)
        plt.plot(predictions,p(predictions),"r--")
        plt.text(0.15, 0.95, 'y='+str(round(z[0],2))+'x+'+str(round(z[1],2)), size=12,horizontalalignment='center',verticalalignment='center',transform=plt.gca().transAxes)
        plt.text(0.15, 0.9, 'R2='+str(round(R2,2)), size=12,horizontalalignment='center',verticalalignment='center',transform=plt.gca().transAxes)
        plt.text(0.15, 0.85, 'MAE='+str(round(MAE,2)), size=12,horizontalalignment='center',verticalalignment='center',transform=plt.gca().transAxes)
        plt.text(0.15, 0.8, 'MAPE='+str(round(MAPE*100,2))+'%', size=12,horizontalalignment='center',verticalalignment='center',transform=plt.gca().transAxes)
        plt.text(0.15, 0.75, 'RMSE='+str(round(RMSE,2)), size=12,horizontalalignment='center',verticalalignment='center',transform=plt.gca().transAxes)
        plt.savefig('./xgboost_results/scatterplot_'+image_features+month+'_'+year+'.png')
        
        
        
        importances =pd.DataFrame(np.resize(model.feature_importances_,
                                    (1,len(x_train.columns))),
                                    columns=x_train.columns)
        #importances.rename(columns ={importances.columns[13]:'Imaging date ks'},inplace=True)
        importances.rename(index={0:image_features+'_'+month+'_'+year}, inplace=True)
        feature_importance=feature_importance.append(importances)
        
        indices = np.array(np.argsort(importances))
        sort=importances.iloc[0,indices[0]]
        
        # bar plots
        '''
        plt.figure(figsize=(4.5, 3))
        plt.title("Feature importances")
        #plt.barh(range(len(importances.columns)), sort,color="r",  align="center")
        plt.barh(range(5), sort[-5:],color="tomato",  align="center")
        for i, v in enumerate(sort[-5:]):
            plt.text(v, i-0.5, str(round(v,2)), color='blue', fontweight='bold')
        # If you want to define your own labels,
        # change sort.index to a list of labels on the following line.
        #plt.yticks(range(x_train.shape[1]), sort.index)
        #plt.ylim([-1, x_train.shape[1]])
        plt.yticks(range(5), sort[-5:].index)
        plt.ylim([-1, 5])
        '''
        # pie plots
        labels = list(sort[-5:].index) 
        labels.append('others')
        sizes = round(sort[-5:],2)*100 
        sizes=pd.concat([sizes,pd.DataFrame([round(1-sum(sort[-5:]),2)*100],index=['others'])]).stack()
        colors = ['red','yellowgreen','lightskyblue','yellow','violet','wheat'] 
        explode = (0,0,0,0,0,0) 
        plt.figure(figsize=(6.5, 3))
        #plt.subplot(121)
        patches,text1,text2 = plt.pie(sizes,
                              explode=explode,
                              #labels=labels,
                              colors=colors,
                              autopct = '%2.0f%%', 
                              shadow = False, 
                              startangle =90, 
                              pctdistance = 0.8) 
        
        #patches
        plt.legend(patches, labels,
                  title=image_features+'_'+month+'_'+year,
                  loc="center left",
                  bbox_to_anchor=(0.7, 0.5, 0, 0))  #x0, y0, width, height

        # x，y
        plt.axis('equal')
        plt.show()        
        
        
        plt.savefig('./xgboost_results/gainImportant_'+image_features+month+'_'+year+'.png')
        plt.close('all')
        #break
        
        # save the best 'target' in the BO
        targets[year+'_'+month][image_features]=target
        
    #r2_j=r2_j+1

# heatmap of the feature_importance
colormap = plt.cm.YlOrRd
mask = np.zeros_like(feature_importance)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(feature_importance.iloc[:-5,:],linewidths=0.1,vmax=0.5,vmin=0.0, 
            square=True, cmap=colormap, linecolor='white', annot=False,
            xticklabels=feature_importance.iloc[:-5,:].columns,
            yticklabels=feature_importance.iloc[:-5,:].index)

#
temp=pd.read_csv('features_important4.csv')

for temp_i in range(45):
    
    importances=temp.loc[temp_i,:]
    indices = np.array(np.argsort(-importances))
    sort=importances.iloc[indices][0:6]
    sort=sort[::-1]
    plt.figure(figsize=(6,5))
    #plt.title("Feature importances")
    plt.barh(range(6), sort,color="r",  align="center")
    for i, v in enumerate(sort):
        plt.text(v, i, str(round(v,2)), color='blue', fontweight='bold', fontsize=14)
    # If you want to define your own labels,
    # change sort.index to a list of labels on the following line.
    plt.yticks(range(6), sort.index,fontsize=14)
    plt.ylim([-1, 6])
    plt.xticks(fontsize=14)
    plt.savefig('./xgboost_results/gainImportant_'+str(temp_i)+'.png')
    plt.close('all')
