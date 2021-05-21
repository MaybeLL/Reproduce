import numpy as np


class k_means():


    def __init__(self,k,dataSet) -> None:
        self.k = k
        self.dataset = dataSet
    def euclDistance(self,vector1, vector2):
        return np.sqrt(np.sum(np.power(vector2 - vector1, 2)))

    def initCentroids(self):
        numSamples, dim = self.dataset.shape
        centroids = np.zeros((self.k, dim))
        for i in range(self.k):
            index = int(np.random.uniform(0, numSamples))
            centroids[i, :] =self.dataset[index, :]
        return centroids
    def run(self):
        numSamples = self.dataset.shape[0]
        # first column stores 所在的类别
        # second column 存残差
        clusterAssment = np.mat(np.zeros((numSamples, 2)))
        clusterChanged = True
    
        ## step 1: init centroids
        centroids = self.initCentroids()
    
        while clusterChanged:
            clusterChanged = False
            ## for each sample
            for i in np.arange(numSamples):
                minDist  = 100000.0
                minIndex = 0
                ## for each centroid
                ## step 2: find the centroid who is closest
                for j in range(self.k):
                    distance = self.euclDistance(centroids[j, :], self.dataset[i, :])
                    if distance < minDist:
                        minDist  = distance
                        minIndex = j
                
                ## step 3: update its cluster
                if clusterAssment[i, 0] != minIndex:
                    clusterChanged = True
                    clusterAssment[i, :] = minIndex, minDist**2
    
            ## step 4: update centroids
            for j in range(self.k):
                pointsInCluster = self.dataset[np.nonzero(clusterAssment[:, 0].A == j)[0]]
                centroids[j, :] = np.mean(pointsInCluster, axis = 0)
    
        print('Congratulations, cluster complete!') 
        return centroids, clusterAssment



if __name__ =="__main__":
    dataSet = []
    fileIn = open('testSet.txt')
    for line in fileIn.readlines():
	    lineArr = line.strip().split('\t')
	    dataSet.append([float(lineArr[0]), float(lineArr[1])])
    dataSet = np.mat(dataSet)
    k_means1 = k_means(3,dataSet)
    centroids, clusterAssment = k_means1.run()
    print(clusterAssment)