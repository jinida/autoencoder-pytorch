import numpy as np
from math import sin, cos, sqrt

''' For MNIST '''
class Distributor:
    def __init__(self, batchSize, dimNum, labelNum):
        if dimNum != 2 or labelNum != 10:
            raise ValueError("the number of labels must be 10 and the number of dimensions must be 2.")
        self.batchSize = batchSize
        self.dimNum = dimNum
        self.labelNum = labelNum
    
    def getSample(self):
        sample = np.empty((self.batchSize, self.dimNum), dtype=np.float32)
        sampleId = np.random.randint(0, self.labelNum, (self.batchSize))
        return sample, sampleId
    
    def setBatchSize(self, batchSize):
        self.batchSize = batchSize

class SwissRollDistributor(Distributor):
    def __init__(self, batchSize, dimNum: int=2, labelNum: int=10) -> None:
        super().__init__(batchSize, dimNum, labelNum)
        
    def getSample(self):
        sample, sampleId = super().getSample()
        
        for batchIdx in range(self.batchSize):
            uniformSample = np.random.uniform(0., 1.) / self.labelNum + sampleId[batchIdx] / self.labelNum
            radius = sqrt(uniformSample) * 3.
            radian = np.pi * 4. * sqrt(uniformSample)
            x = radius * cos(radian)
            y = radius * sin(radian)
            sample[batchIdx] = np.array([x, y]).reshape((2, ))
            
        return sample, sampleId
    
class UniformDistributor(Distributor):
    def __init__(self, batchSize, dimNum: int=2, labelNum: int=10, minValue=-1, maxValue=1):
        super().__init__(batchSize, dimNum, labelNum)
        self.minValue = minValue
        self.maxValue = maxValue
        
    def getSample(self):
        sample, sampleId = super().getSample()
        for batchIdx in range(self.batchSize):
            gridNum = int(np.ceil(np.sqrt(self.labelNum)))
            gridSize = (self.maxValue - self.minValue) / gridNum
            x, y = np.random.uniform(-gridSize / 2, gridSize / 2, (2, ))
            label = self.labelIndices[batchIdx]
            i = label / gridNum
            j = label % gridNum
            x += j * gridSize + self.minValue + 0.5 * gridSize
            y += i * gridSize + self.minValue + 0.5 * gridSize
            sample[batchIdx] = np.array([x, y]).reshape((2, ))
            
        return sample, sampleId
            
class GaussianDistributor(Distributor):
    def __init__(self, batchSize, dimNum: int=2, labelNum: int=10, mean: int=0, varience: int=1):
        super().__init__(batchSize, dimNum, labelNum)
        self.mean = mean
        self.varience = varience
    
    def getSample(self):
        sample, sampleId = super().getSample()
        for batchIdx in range (self.batchSize):
            x, y = np.random.normal(self.mean, self.varience, (2, ))
            angle = np.angle((x - self.mean) + 1j * (y - self.mean), deg=True)
            dist = np.sqrt((x - self.mean) ** 2 + (y - self.mean) ** 2)
            label  = 0 if dist < 1. else int(angle * (self.labelNum - 1)) // 360
            if label < 0:
                label = label + self.labelNum - 1
            label += 1
            sample[batchIdx], sampleId[batchIdx] = np.array([x, y]).reshape((2, )), label
            
        return sample, sampleId
        

class GaussianMixtureDistributor(Distributor):
    def __init__(self, batchSize, dimNum: int=2, labelNum: int=10, xVarience: float=0.5, yVarience: float=0.1) -> None:
        super().__init__(batchSize, dimNum, labelNum)
        self.xVarience = xVarience
        self.yVarience = yVarience
        self.shift = 1.4
    
    def getSample(self):
        sample, sampleId = super().getSample()
        x = np.random.normal(0, self.xVarience, (self.batchSize, 1))
        y = np.random.normal(0, self.yVarience, (self.batchSize, 1))
        sample = np.empty((self.batchSize, self.dimNum), dtype=np.float32)
        for batchIdx in range (self.batchSize):
            r = 2. * np.pi / self.labelNum * sampleId[batchIdx]
            newX = x[batchIdx] * cos(r) - y[batchIdx] * sin(r) + self.shift * cos(r)
            newY = x[batchIdx] * sin(r) + y[batchIdx] * cos(r) + self.shift * sin(r)
            sample[batchIdx] = np.array([newX, newY]).reshape((2, ))
            
        return sample, sampleId