import numpy as np
import math

class linreg:
    def __init__(self,x,y):
        self.xdata = x
        self.ydata = y
    def testtrainsplit(self,splitratio):
        endidx = math.floor(self.xdata.shape[0]*(1 - splitratio))
        self.xtrain = self.xdata.iloc[0:endidx,:]
        self.xtest = self.xdata.iloc[endidx:self.xdata.shape[0],:]
        self.ytrain = self.ydata.iloc[0:endidx,:]
        self.ytest = self.ydata.iloc[endidx:self.xdata.shape[0],:]
        return self.xtrain,self.xtest,self.ytrain,self.ytest
    def findmean(self,xdata):
        #this is a pandas series
        self.meanval = self.xdata.mean(axis = 0)
        return self.meanval
    def findstd(self,xdata):
        #this is a pandas series
        self.stdval = self.xdata.std(axis = 0)
        return self.stdval
    def normalize(self,xdata):
        meanvl = self.findmean(xdata)
        stdvl = self.findstd(xdata)
        self.normalizeddata = (xdata.subtract(meanvl,axis = 1)).div(stdvl)
        return self.normalizeddata
    def graddescent(self,xdata,ydata,max_iters = 1500,alpha = 0.01):
        #the xdata should already be scaled using the "normalize" function above
        #the ydata should be the ydata for training
        
        #turns the previous dataframe into a numpy array
        self.xdatamatrix = xdata.to_numpy()
        
        #array of bias values
        self.biascolumn = np.ones((xdata.shape[0],1))
        
        #self.trainingxdata includes the bias value included into the dataset
        self.trainingxdata = np.append(self.biascolumn,self.xdatamatrix,axis = 1)
        
        #theta values to start gradient descent
        self.theta = np.zeros((self.trainingxdata.shape[1],1))
        
        #a matrix to add the cost of each iteration
        iternum =np.arange(max_iters).reshape(max_iters,1)
        costtemplate = np.zeros((max_iters,1))
        self.itermatrix = np.append(iternum,costtemplate, axis = 1)
        
        for i in range(0,max_iters):
            #multiplying the x variables with the theta values
            h = np.dot(self.trainingxdata,self.theta)
            #number of training samples 
            m = xdata.shape[0]
            #calculating the cost
            cost = (1/(2*m))*np.dot((np.transpose(h - ydata)),(h - ydata))
            self.itermatrix[i,1] = cost
            #starting gradient descent work
            self.theta[0,:] = self.theta[0,:] - (alpha/m)*(np.dot(np.transpose(self.trainingxdata[:,0]),(h - ydata)))
            self.theta[1:,:] = self.theta[1:,:] - (alpha/m)*(np.dot(np.transpose(self.trainingxdata[:,1:]),(h - ydata)))
        
        #calculating the meansquareerror
        hpredicted = np.dot(self.trainingxdata,self.theta)
        squareerror = np.dot(np.transpose(hpredicted - ydata),(hpredicted - ydata))
        self.meansquareerror = squareerror/m
        return self.itermatrix, self.theta,self.meansquareerror
    def predict(self,x_testdata):
        #turn test dataframe into a numpy array
        self.xtestmatrix = x_testdata.to_numpy()
        
        #adding bias column
        self.testingxdata = np.append(np.ones((x_testdata.shape[0],1)),self.xtestmatrix,axis = 1)
        
        self.ypredicttest = np.dot(self.testingxdata,self.theta)
        
        return self.ypredicttest