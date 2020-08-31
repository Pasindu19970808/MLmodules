import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import operator
from  timeit import default_timer as timer
dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]

#[[plt.scatter(x = ii[0], y = ii[1], color = i) \
 #for ii in dataset[i]]for i in dataset]


class knneighbors:
    def __init__(self,k_number = 3):
        self.knumber = k_number
    def fit(self,data,predict):
        if len(data) >= self.knumber:
            #cannot have k less than number of voting groups
            warnings.warn('K is less than number of voting groups')
        
        distances = []
        for group in data:
            for features in data[group]:
                #calculate difference between points
                #np.sqrt(np.sum(np.square\
                #(np.array(dataset['k']) - np.array(new_features)),axis = 1))
                difference = np.linalg.norm(np.array(features) - np.array(predict))
                #[difference,group] looks like [2.5,'k']
                distances.append([difference,group])
        #operator.itemgetter indicates that we want to sort by the first entry in list
        votes = [i[1] for i in sorted(distances, key = operator.itemgetter(0))[:self.knumber]]
        
        #this is a dictionary
        votecount = Counter(votes)
        #classification result
        
        classifiedgroup = max(votecount.items(),key = operator.itemgetter(1))[0]
        #classifiedgroup = votecount.most_common(1)[0][0]
        return classifiedgroup
        
        
    
        
        
        #votes = [i[1] for i in ]


knn = knneighbors()
dist = knn.fit(dataset,new_features)