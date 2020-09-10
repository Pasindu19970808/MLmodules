import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random



class kmeansclustering:
    def __init__(self,k_iterations,k_number = 4, init_method = 'Normal'):
        self.knumber = k_number
        self.kiterations = k_iterations
        self.initmethod = init_method
    def shuffle(self,data):
        #maintain the dataframe format while shuffling
        shuffleddata = data.sample(frac = 1)
        return shuffleddata
    def scale(self,data):
        #scaling the data
        meanarray = np.mean(data,axis = 0)
        stdarray = np.std(data, axis = 0)
        #still a dataframe
        scaleddata = (data - meanarray)/stdarray
        return scaleddata

    def runkmeans(self,data,normalize = 'False'):
        if normalize == 'True':
            #scaled dataframe
            self.clusteringdataframe = self.scale(data)
        else:
            self.clusteringdataframe = data
            
        dflist = self.clusteringdataframe.astype(float).values.tolist()   
        
        if self.initmethod == 'Normal':
            #initial centroid indices
            ind = np.random.permutation(len(df))[:self.knumber].tolist()
            centroids = [dflist[i] for i in ind]
        #turn the dataframe into a list
        #we do this to easily loop through samples
        
        #make a dictionary of clusters
        clusters = {}
        for i in range(self.knumber):
            clusters[i] = []
        
        #list to keep and update cluster index
        clusterindex = [np.nan]*len(dflist)
        for iteration in range(0,self.kiterations):
            j = 0
            for trainingsample in dflist:
                euclidean_distancearray = np.linalg.norm((np.array(centroids) - np.array(trainingsample)),axis = 1)
                #centroid index with lowest distance to sample
                minimumclusterindex = np.argmin(euclidean_distancearray)
                #continuously update the cluster index 
                clusterindex[j] = minimumclusterindex
                j += 1
                clusters[minimumclusterindex].append(trainingsample)
            for group in clusters:
                centroids[group] = np.mean(clusters[group],axis = 0).tolist()
            
        #create clustered dataframe
        columnlist = data.columns.tolist()
        columnlist.append('Cluster Index')
        clusteredarray = np.concatenate((data.to_numpy(),np.array(clusterindex).reshape(-1,1)),axis = 1)
        clustereddf = pd.DataFrame(clusteredarray, columns = columnlist)
        return clustereddf
'''Has 
    def visualize(self,clustered_dataframe):
        fig = plt.Figure(figsize = (10,10))
        ax1 = fig.add_subplot(111)
        column_names = clustered_dataframe.columns.tolist()
        sns.scatterplot(x = column_names[0],y = column_names[1], hue = column_names[2], data = clustered_dataframe,ax = ax1)
        
'''
        

            
            
            
            
            
            
            
            
            
            
            
            
            
filename = r'Mall_Customers.csv'
df = pd.read_csv(filename)

kmeans = kmeansclustering(k_iterations = 10,k_number = 5)     
resultdf = kmeans.runkmeans(kmeans.shuffle(df),normalize = 'True')
kmeans.visualize(resultdf)