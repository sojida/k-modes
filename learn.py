#Import all the necessary packages

import pandas as pd
import numpy as np

import matplotlib.pylab as plt

import seaborn as sns


#to scale the data using z-score 
from sklearn.preprocessing import StandardScaler

#importing clustering algorithms
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids
from kmodes.kmodes import KModes 
import pickle

from sklearn.metrics import silhouette_score

data = pd.read_excel('data.xlsx')


# remove variables that are not required for our analysis
data.drop(columns = ['Sl_No', 'Customer Key'], inplace = True)

# remove duplicated data
data=data[~data.duplicated()]

# scale data
scaler = StandardScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

def WCSS(data):
    # WCSS is the sum of the variance between the observations in each cluster. 
    WCSS = {} 

    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(data)
        WCSS[k] = kmeans.inertia_

    plt.figure()
    plt.plot(list(WCSS.keys()), list(WCSS.values()), 'bx-')
    plt.xlabel("Number of cluster")
    plt.ylabel("WCSS")
    plt.show()
    
# silhouette score
def SC(data):
    sc = {} 

    # iterate for a range of Ks and fit the scaled data to the algorithm. Store the Silhouette score for that k 
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=1).fit(data)
        labels = kmeans.predict(data)
        sc[k] = silhouette_score(data, labels)

    #Elbow plot
    plt.figure()
    plt.plot(list(sc.keys()), list(sc.values()), 'bx-')
    plt.xlabel("Number of cluster")
    plt.ylabel("Silhouette Score")
    plt.show()
    
# SC(scaled_data)
    
def EC_KModes(data):
    cost = [] 
    K = range(1,5) 
    for k in list(K): 
        kmode = KModes(n_clusters=k, init = "random", n_init = 5, verbose=1) 
        kmode.fit_predict(data) 
        cost.append(kmode.cost_) 
        
    plt.plot(K, cost, 'x-') 
    plt.xlabel('No. of clusters') 
    plt.ylabel('Cost') 
    plt.title('Elbow Curve') 
    plt.show()
    
# EC_KModes(scaled_data)


def kmode_model():
    kmode = KModes(n_clusters=3, random_state=1, max_iter=1000) 
    clusters = kmode.fit_predict(scaled_data) # [0, 1, 2]
    data['Cluster'] = clusters # [avg_credit_limit, ..., cluster]
    
    print(data.head())

    mode = data.groupby('Cluster').apply(lambda x: x.mode().iloc[0])

    dataframe_mode = pd.concat([mode.iloc[:, :-1]], axis=0)

    dataframe_mode.index = ['Cluster 1 Mode', 'Cluster 2 Mode', 'Cluster 3 Mode']
    print(dataframe_mode.T)
    # save the model to disk
    with open('kmode.pickle', 'wb') as f:
        pickle.dump(kmode, f)
    
# kmode_model()

def kmeans_model():
    kmeans = KMeans(n_clusters=3, max_iter= 1000, random_state=1)
    kmeans.fit(scaled_data)

    #Adding predicted clusters to the original data and scaled data 
    data['Cluster'] = kmeans.predict(scaled_data)
    mean = data.groupby('Cluster').mean()
    
    df_kmeans = pd.concat([mean], axis=0)
    df_kmeans.index = ['Cluster 1 Mean', 'Cluster 2 Mean', 'Cluster 3 Mean']
    print(df_kmeans.T)
    print(data.head())
    # save the model to disk
    with open('kmeans.pickle', 'wb') as f:
        pickle.dump(kmeans, f)
    
kmode_model()