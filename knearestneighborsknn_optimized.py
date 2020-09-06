import numpy as np 
from collections import Counter
import warnings
import operator
from  timeit import default_timer as timer
import random
import pandas as pd
#[[plt.scatter(x = ii[0], y = ii[1], color = i) \
 #for ii in dataset[i]]for i in dataset]


class knneighbors:
    #it is always better to have a odd k number such that there will never be a 50 50 split in the votes
    def __init__(self,k_number = 3):
        self.knumber = k_number
    def fit(self,data,predict):
        if len(data) >= self.knumber:
            #cannot have k less than number of voting groups
            warnings.warn('K is less than number of voting groups')
        
        distances = []
        for group in data:
            #for features in data[group]:
                #calculate difference between points
                #np.sqrt(np.sum(np.square\
                #(np.array(dataset['k']) - np.array(new_features)),axis = 1))
            #find eucledian distance between the points
            difference = np.linalg.norm((np.array(data[group]) - np.array(predict)),axis = 1)
                #[difference,group] looks like [2.5,'k']
            #this is an array of the group that is under processing
            grouparray = np.array([group]*difference.shape[0])
            templist = np.concatenate((difference.reshape(-1,1),grouparray.reshape(-1,1)),axis = 1).tolist()
            
            #[distances.append(i) for i in templist]
            distances.extend(templist)
        #operator.itemgetter indicates that we want to sort by the first entry in list
        #sorted(distances, key = operator.itemgetter(0))[:self.knumber] gives a list with 
        #the sorted euclidean distances. [:self.knumber] gives us the lowest 5 distances
        votes = [i[1] for i in sorted(distances, key = operator.itemgetter(0))[:self.knumber]]
        
        #this is a dictionary
        votecount = Counter(votes)
        #classification result
        
        classifiedgroup = max(votecount.items(),key = operator.itemgetter(1))[0]
        #classifiedgroup = votecount.most_common(1)[0][0]
        return classifiedgroup
        
        
starttime = timer()      
df = pd.read_csv('data.csv')
df.drop('id',axis = 1, inplace = True)
df['class'] = np.nan
#manual label encoding
df.loc[df['diagnosis'] == 'M','class'] = 4
df.loc[df['diagnosis'] == 'B','class'] = 2
dffinal = df.drop('diagnosis',axis = 1)
#we have converted all the data into a list 
#where each row is a sample
#sometimes data can come as quotes(string)
#better to convert to float always
full_data = dffinal.astype(float).values.tolist() 
#now we shuffle the data. 
#by converting to list and having every sample(row)
#as a element in the list, the relationship between
#variables and output is maintained. 
random.shuffle(full_data)

#creating a train test split
test_size = 0.2
train_set = {2:[],4:[]}
test_set = {2:[],4:[]}
#upto the last 20% of data
train_data = full_data[:-int(test_size*len(full_data))]
#from the last 20% of data
test_data = full_data[-int(test_size*len(full_data)):]

#populate to a dictionary
for i in train_data:
    train_set[i[-1]].append(i[:-1])
for i in test_data:
    test_set[i[-1]].append(i[:-1])


knn = knneighbors(k_number = 4)

correct = 0
total = 0

for group in test_set:
    #passing one sample at a time
    for data in test_set[group]:
        vote = knn.fit(train_set,data)
        if group == vote:
            correct += 1
        total +=1
print('Accuracy:',correct/total)
endtime = timer()
print(endtime - starttime)