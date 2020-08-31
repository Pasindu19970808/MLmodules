import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\ASUS\Desktop\Python code\Data Science\Mall_Customers.csv')
dimensions = len(df.columns)
#number of clusters
K = 5
numiter = 10
#amount of rows in df
rownum = len(df)
#attaching index column to df
df['Cluster Index'] = [0]*rownum
newdimension = len(df.columns)

fig0 = plt.figure()
ax2 = fig0.add_subplot(121)
x = df[df.columns.values.tolist()[0]]
y = df[df.columns.values.tolist()[1]]
ax2.scatter(x,y, marker = '*')
ax2.set_xlabel(df.columns.values.tolist()[0])
ax2.set_ylabel(df.columns.values.tolist()[1])


#random rows to extract for initial centroids
positions = np.random.permutation(rownum).tolist()[:K]
#initial centroids
initcentroids = df.iloc[positions,list(range(dimensions))]
initcentroids = initcentroids.reset_index(drop=True)
#data = {'x1':[3,6,8],'x2':[3,2,5]}
#initcentroids = pd.DataFrame(data = data, columns = ['x1','x2'])
        
for i in range(0,numiter):
    for j in range(0,rownum):
        #list to hold the root mean squares
        rms = [0]*K
        for h in range(0,K):
            #distance between point under consideration and centroid
            dist = df.iloc[j,list(range(0,dimensions))] - initcentroids.iloc[h,list(range(0,dimensions))]
            dist = sum(dist**2)
            rms[h] = dist
        minpos = rms.index(min(rms))
        df.iloc[j,-1] = minpos
    prevcentroid = initcentroids
    #resetting the initial centroids
    for k in range(0,K):
        initcentroids.iloc[k,:] = df.loc[lambda df:df['Cluster Index'] == k].iloc[:,[0,1]].mean(axis = 0)


ax1 = fig0.add_subplot(122)
cmapcols = plt.get_cmap('gist_rainbow')
colnum = np.linspace(0,1,len(range(0,K)))
colors = iter(cmapcols(colnum))
for k in range(0,K):
    x = df.loc[lambda df:df['Cluster Index'] == k].iloc[:,0]
    y = df.loc[lambda df:df['Cluster Index'] == k].iloc[:,1]
    ax1.scatter(x,y,marker = '*',label = k,color = next(colors))
    
ax1.set_xlabel(df.columns.values.tolist()[0])
ax1.set_ylabel(df.columns.values.tolist()[1])
ax1.legend()
    
    
    
        
        
            
        
            
            
            
            
            
