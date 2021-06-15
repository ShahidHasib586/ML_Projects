# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 23:18:17 2021

@author: shahi
"""


# Principal Component Analysis (PCA)

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Part 1 - Data Preprocessing

# Importing the dataset
df = pd.read_csv('M_Data analysis.csv')
df.head(10)
# Taking care of missing data
# Updated Imputer


Basic_statistics = df.iloc[:,1:].describe()

for c in df.columns[1:]:
    df.boxplot(c,by='Brand_Encoded',figsize=(14,8),fontsize=14)
    plt.title("{}\n".format(c),fontsize=16)
    plt.xlabel("Vehicle data", fontsize=16)

#plot co-varience matrix (PriceEuro vs Range)
plt.figure(figsize=(20,12))
plt.scatter(df['PriceEuro'],df['Range_Km'],c=df['Brand_Encoded'],edgecolors='pink',alpha=1.5,s=300)
plt.grid(True)
plt.title("Scatter plot of two features showing the \ncorrelation and class seperation",fontsize=15)
plt.xlabel("PriceEuro",fontsize=15)
plt.ylabel("Range_Km'",fontsize=15)
plt.show()
#plot co-varience matrix (Efficiency_WhKm vs Range)
plt.figure(figsize=(20,12))
plt.scatter(df['Efficiency_WhKm'],df['Range_Km'],c=df['Brand_Encoded'],edgecolors='pink',alpha=1.5,s=300)
plt.grid(True)
plt.title("Scatter plot of two features showing the \ncorrelation and class seperation",fontsize=15)
plt.xlabel("Efficiency_WhKm",fontsize=15)
plt.ylabel("Range_Km'",fontsize=15)
plt.show()
#plot co-varience matrix (AccelSec vs Range)
plt.figure(figsize=(20,12))
plt.scatter(df['AccelSec'],df['Range_Km'],c=df['Brand_Encoded'],edgecolors='pink',alpha=1.5,s=300)
plt.grid(True)
plt.title("Scatter plot of two features showing the \ncorrelation and class seperation",fontsize=15)
plt.xlabel("AccelSec",fontsize=15)
plt.ylabel("Range_Km'",fontsize=15)
plt.show()
#plot co-varience matrix (TopSpeed_KmH vs Range)
plt.figure(figsize=(20,12))
plt.scatter(df['TopSpeed_KmH'],df['Range_Km'],c=df['Brand_Encoded'],edgecolors='pink',alpha=1.5,s=300)
plt.grid(True)
plt.title("Scatter plot of two features showing the \ncorrelation and class seperation",fontsize=15)
plt.xlabel("TopSpeed_KmH",fontsize=15)
plt.ylabel("Range_Km'",fontsize=15)
plt.show()
#plot co-varience matrix (Battery_Pack Kwh vs Range)
plt.figure(figsize=(20,12))
plt.scatter(df['Battery_Pack Kwh'],df['Range_Km'],c=df['Brand_Encoded'],edgecolors='pink',alpha=1.5,s=300)
plt.grid(True)
plt.title("Scatter plot of two features showing the \ncorrelation and class seperation",fontsize=15)
plt.xlabel("Battery_Pack Kwh",fontsize=15)
plt.ylabel("Range_Km'",fontsize=15)
plt.show()
#plot co-varience matrix (FastCharge_KmH vs Range)
plt.figure(figsize=(20,12))
plt.scatter(df['FastCharge_KmH'],df['Range_Km'],c=df['Brand_Encoded'],edgecolors='pink',alpha=1.5,s=300)
plt.grid(True)
plt.title("Scatter plot of two features showing the \ncorrelation and class seperation",fontsize=15)
plt.xlabel("FastCharge_KmH",fontsize=15)
plt.ylabel("Range_Km'",fontsize=15)
plt.show()

def correlation_matrix(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure(figsize=(16,12))
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('EV dataset feature co-relation',fontsize=15)
    labels=df.columns
    ax1.set_xticklabels(labels,fontsize=9)
    ax1.set_yticklabels(labels,fontsize=9)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[0.1*i for i in range(-11,11)])
    plt.show()

correlation_matrix(df)

#PCA  data processing

from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()


X = df.drop('Brand_Encoded',axis=1)
y = df['Brand_Encoded']

X = scaler.fit_transform(X)

dfx = pd.DataFrame(data=X,columns=df.columns[1:])

dfx.head(10)


dfx.describe()

#PCA class import and analysis

from sklearn.decomposition import PCA

pca = PCA(n_components=None)
    
   
dfx_pca = pca.fit(dfx)

 
plt.figure(figsize=(10,6))
plt.scatter(x=[i+1 for i in range(len(dfx_pca.explained_variance_ratio_))],
            y=dfx_pca.explained_variance_ratio_,
           s=200, alpha=0.75,c='orange',edgecolor='k')
plt.grid(True)
plt.title("Explained variance ratio of the \nfitted principal component vector\n",fontsize=25)
plt.xlabel("Principal components",fontsize=15)
plt.xticks([i+1 for i in range(len(dfx_pca.explained_variance_ratio_))],fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("Explained variance ratio",fontsize=15)
plt.show()

'''
The above plot means that the $1^{st}$ principal component explains about 65% of the total variance in the data and the $2^{nd}$ component explians further 14%. Therefore, if we just consider first two components, they together explain 79% of the total variance.
'''

'''
Showing better class separation using principal components
Transform the scaled data set using the fitted PCA object
'''


dfx_trans = pca.transform(dfx)

#Put it in a data frame

dfx_trans = pd.DataFrame(data=dfx_trans)
dfx_trans.head(10)

'''
Plot the first two columns of this transformed data set with the color set to original ground truth class label
'''

plt.figure(figsize=(40,24))
plt.scatter(dfx_trans[0],dfx_trans[1],c=df['Brand_Encoded'],edgecolors='k',alpha=0.75,s=150)
plt.grid(True)
plt.title("Class separation using first two principal components\n",fontsize=20)
plt.xlabel("Principal component-1",fontsize=15)
plt.ylabel("Principal component-2",fontsize=15)
plt.show()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    