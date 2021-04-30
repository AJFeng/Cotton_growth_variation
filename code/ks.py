# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 18:43:20 2020

@author: af3bd
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ipdb

soil=pd.read_csv('soil features downsample.csv')
weather=pd.read_csv('weather feature 2019.csv')
irrigation_data_2019=pd.read_csv('irrigation data 2019.csv')

soil=pd.read_csv('soil features downsample W.csv')
weather=pd.read_csv('weather feature 2018.csv')
irrigation_data_2018=pd.read_csv('irrigation data 2018.csv')


Dri_start=np.zeros([len(weather),1])
ks=np.zeros([len(weather),1])
ET_adj=np.zeros([len(weather),1])
Dri_end=np.zeros([len(weather),1])
raw=np.zeros([len(weather),1])
irrigation=0


result=np.zeros([len(soil),len(weather)])



for j in range(len(soil)):
    
    for i in range(len(Dri_start)):
        flag=0
        
        # irrigation
        irrigation=0
        '''
        # 2019 as following, did not use DAP but use month+date
        if weather.iloc[i,0]==8:   # check if irrigation is applied
            if (weather.iloc[i,1]==6) and (soil.iloc[j,21]==3) and (soil.iloc[j,22]==4):
                flag=1
                irrigation=22
            if (weather.iloc[i,1]==14) and (((soil.iloc[j,21]==3) and (soil.iloc[j,22]==4)) or 
               ((soil.iloc[j,21]==1) and (soil.iloc[j,22]==5)) or
               ((soil.iloc[j,21]==1) and (soil.iloc[j,22]==2))):
                flag=1
                irrigation=22
            if (weather.iloc[i,1]==19) and (((soil.iloc[j,21]==3) and (soil.iloc[j,22]==4)) or 
               ((soil.iloc[j,21]==1) and (soil.iloc[j,22]==2))):
                flag=1
                irrigation=22
            if (weather.iloc[i,1]==19) and (((soil.iloc[j,21]==2) and (soil.iloc[j,22]==3))):
                flag=1
                irrigation=6.3
        '''
        
        # 2018 as following, use DAP
        if i<=(int(irrigation_data_2018.columns[-1])-1):
            irrigation_count=np.argwhere(((np.array(irrigation_data_2018.columns[2:])).astype(np.int)-1)>=i)[0,0]+2
        else:
            irrigation_count=len(irrigation_data_2018.columns)-1
        
        
        if (irrigation_count<len(irrigation_data_2018.columns)) and (weather.iloc[i,3]==int(irrigation_data_2018.columns[irrigation_count])-1)\
        and (irrigation_data_2018[irrigation_data_2018['irrigation_length'].isin([soil.iloc[j,21]]) &\
            irrigation_data_2018['irrigation_circle'].isin([soil.iloc[j,22]])].iloc[:,irrigation_count].item()!=0):
            flag=1
            irrigation=irrigation_data_2018[irrigation_data_2018['irrigation_length'].isin([soil.iloc[j,21]]) &\
            irrigation_data_2018['irrigation_circle'].isin([soil.iloc[j,22]])].iloc[:,irrigation_count].item()
            
            #print(irrigation)
        
        
        #if (weather.iloc[i,4]>10) or (i == 0) or (flag==1):  # if rain>10 or the first day or irrigation applied
        if i == 0:  # the first day
            Dri_start[i]=0
            ks[i]=1
        else:
            Dri_start[i]=Dri_end[i-1]
            
        raw[i]=soil.iloc[j,20]*weather.iloc[i,7] # RAW×Z_r
        
        '''
        if (i!=0) and (flag!=1) and (weather.iloc[i,4]<10):
            if Dri_start[i]<raw[i]:
                ks[i]=1
            else:
                taw=soil.iloc[j,19]*weather.iloc[i,7] # TAW×Z_r
                ks[i]=(taw-Dri_start[i])/(taw-raw[i])
        '''        
        taw=soil.iloc[j,19]*weather.iloc[i,7] # TAW×Z_r
        if Dri_start[i]<raw[i]:
            ks[i]=1
        else:
            ks[i]=(taw-Dri_start[i])/(taw-raw[i])
        
        ET_adj[i]=weather.iloc[i,6]*ks[i] # ET_adj[i]=ET_c*Ks
        remain=taw-Dri_start[i]
        if (weather.iloc[i,4]+irrigation+remain)<taw:
            refill=weather.iloc[i,4]+irrigation
        else:
            refill=taw-remain
        Dri_end[i]=Dri_start[i]+ET_adj[i]-refill  # D_(r,i)=D_(r,i-1)+ET_(c,i)-P_i-I_i, unit:mm 
    
    result[j,:]=ks.T

min_Ks=np.min(result,axis=0)

# plot the ks map of a specific date
#col_name=pd.read_csv('ks_irrigation_2019_col_name.csv')
col_name=pd.read_csv('ks_irrigation_2018_col_name.csv')
result_with_name=pd.DataFrame(result,columns=col_name.columns)
#np.savetxt('ks_irrigation_2019.csv', result, delimiter=',')
#result_with_name.to_csv('ks_irrigation_2019.csv',index=False)   
result_with_name.to_csv('ks_irrigation_2018.csv',index=False)   

    
ks=pd.read_csv('ks_irrigation_2019.csv')
temp=np.vstack((soil.iloc[:,2],soil.iloc[:,3],ks['7/29/2019'])).T
ax=sns.scatterplot(x=temp[:,0],y=temp[:,1],hue=temp[:,2],palette=plt.cm.jet) 
#ax.legend(bbox_to_anchor=(0.5, 0.95))     
# control x and y limits
#plt.ylim(36.4098008, 36.4127458)   #2019
#plt.xlim(-89.696419, -89.694718)

plt.ylim(36.4098008, 36.4127458)   #2018
plt.xlim(-89.698419, -89.696718)

temp2=np.resize(ks['8/11/2019'],(63,38))
plt.imshow(temp2, cmap='jet_r',vmin=0, vmax=1)
plt.axis('off')
plt.colorbar()

img08=pd.read_csv('data_08 downsample.csv').drop(['row','col','longitude','latitude','GRVI', 'EXG', 'EXG_EXR','H','Thermal','Lab_a','canopy_size_raw'],axis=1)
temp3=np.resize(img08['NDVI'],(63,38))
a=pd.DataFrame(ks.min(axis=0))
ax=sns.scatterplot(a.index,a.iloc[:,0])


# compare the ks and the soil moisture sensors
index=[]  #s5
for j in range(len(soil)):
    if ((soil.iloc[j,0]>=9) and (soil.iloc[j,0]<=12)) and ((soil.iloc[j,1]>=2) and (soil.iloc[j,1]<=5)):
        print(j)
        index.append(j)
     
temp=np.zeros([len(index),119])
i=0
for j in index:
    temp[i,:]=ks.iloc[j,:]
    i=i+1
        
index=[] #s4
for j in range(len(soil)):
    if ((soil.iloc[j,0]>=9) and (soil.iloc[j,0]<=12)) and ((soil.iloc[j,1]>=22) and (soil.iloc[j,1]<=25)):
        print(j)
        index.append(j)

temp=np.zeros([len(index),119])
i=0
for j in index:
    temp[i,:]=ks.iloc[j,:]
    i=i+1
    
index=[] #s3
for j in range(len(soil)):
    if ((soil.iloc[j,0]>=21) and (soil.iloc[j,0]<=24)) and ((soil.iloc[j,1]>=31) and (soil.iloc[j,1]<=34)):
        print(j)
        index.append(j)

temp=np.zeros([len(index),119])
i=0
for j in index:
    temp[i,:]=ks.iloc[j,:]
    i=i+1
    
index=[] #s2
for j in range(len(soil)):
    if ((soil.iloc[j,0]>=21) and (soil.iloc[j,0]<=24)) and ((soil.iloc[j,1]>=55) and (soil.iloc[j,1]<=58)):
        print(j)
        index.append(j)

temp=np.zeros([len(index),119])
i=0
for j in index:
    temp[i,:]=ks.iloc[j,:]
    i=i+1
    
index=[] #s1
for j in range(len(soil)):
    if ((soil.iloc[j,0]>=3) and (soil.iloc[j,0]<=6)) and ((soil.iloc[j,1]>=44) and (soil.iloc[j,1]<=47)):
        print(j)
        index.append(j)

temp=np.zeros([len(index),119])
i=0
for j in index:
    temp[i,:]=ks.iloc[j,:]
    i=i+1



# calculate the ks features
ks=pd.read_csv('ks_irrigation_2019.csv')
date=len(ks.columns)-1    # for harvest
#ks=ks.iloc[:,0:56] # 56 for July, 89 for Aug,  111 for Sep   2019
#date=120 # 42 for June, 61 for July, 96 for Aug, 120 for Sep   2018    #79 for Aug   2017

features=np.zeros([len(ks),18])
features[:,0]=np.transpose(ks.iloc[:,date])
ks=ks.iloc[:,0:date+1]


for i in range(len(ks)):
    features[i,1]=ks.iloc[i,:].map(lambda x: x < 1).sum()
for i in range(len(ks)):
    features[i,3]=ks.iloc[i,:].map(lambda x: 0.9<=x < 1).sum()
for i in range(len(ks)):
    features[i,5]=ks.iloc[i,:].map(lambda x: 0.8<=x < 0.9).sum()
for i in range(len(ks)):
    features[i,7]=ks.iloc[i,:].map(lambda x: 0.7<=x < 0.8).sum()
for i in range(len(ks)):
    features[i,9]=ks.iloc[i,:].map(lambda x: 0.6<=x < 0.7).sum()
for i in range(len(ks)):
    features[i,11]=ks.iloc[i,:].map(lambda x: 0.5<=x < 0.6).sum()
for i in range(len(ks)):
    features[i,13]=ks.iloc[i,:].map(lambda x: 0.4<=x < 0.5).sum()
for i in range(len(ks)):
    features[i,15]=ks.iloc[i,:].map(lambda x: x < 0.4).sum()


for i in range(len(ks)):
    longest=0
    day=0
    for j in range(len(ks.columns)):
        if ks.iloc[i,j]>=1:
            day=0
        else:
            day=day+1
            if day>longest:
                longest=day
    features[i,2]=longest
    
def Largest_day_number(feature_id, threshold,ks,features):
    for i in range(len(ks)):
        longest=0
        day=0
        for j in range(len(ks.columns)):
            if ks.iloc[i,j]>threshold:
                day=0
            else:
                day=day+1
                if day>longest:
                    longest=day
        features[i,feature_id]=longest
    return features

features=Largest_day_number(4, 0.9,ks,features)
features=Largest_day_number(6, 0.8,ks,features)
features=Largest_day_number(8, 0.7,ks,features)
features=Largest_day_number(10, 0.6,ks,features)
features=Largest_day_number(12, 0.5,ks,features)
features=Largest_day_number(14, 0.4,ks,features)
features=Largest_day_number(16, 0.3,ks,features)

for i in range(len(ks)):
    date=0
    for j in range(len(ks.columns)):
        date=date+1
        if ks.iloc[i,j]<1:
            features[i,17]=date
            date=0
            break

col_name=[ks.columns[-1]+' ks',	'Total days ks<1','Largest day number ks<1','Total days 0.9<=ks<1',
          'Largest day number ks<0.9','Total days 0.8<=ks<0.9',
          'Largest day number ks<0.8	','Total days 0.7<=ks<0.8',
          'Largest day number ks<0.7','Total days 0.6<=ks<0.7',
          'Largest day number ks<0.6	','Total days 0.5<=ks<0.6',
          'Largest day number ks<0.5','Total days 0.4<=ks<0.5',
          'Largest day number ks<0.4','Total days <0.4',
          'Largest day number ks<0.3',	'date start to ks<1']
features_with_name=pd.DataFrame(features,columns=col_name)
#features_with_name.to_csv('ks features 092018.csv',index=False)
features_with_name.to_csv('ks features 2019 harvest.csv',index=False)

#np.savetxt('ks features 072019.csv', features, delimiter=',')
#np.savetxt('ks features 2017 harvest.csv', features, delimiter=',')

