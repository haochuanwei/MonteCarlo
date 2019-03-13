import numpy as np
from abc import ABC, abstractmethod
from speedup import BatchRandomNumber

class Distribution(object):
    def __init__(self, dimension, densityFunction):
        self.validate(dimension, densityFunction)
        self.dimension = dimension
        self.densityFunction = densityFunction
        
    def validate(self, dimension, densityFunction):
        assert type(dimension) is tuple
        assert bool(densityFunction(np.random.random(dimension))+1e-8)
        
    def density(self, npArr):
        assert npArr.shape == self.dimension
        return self.densityFunction(npArr)

class Metropolis(ABC):
    def __init__(self, distObject):
        assert type(distObject) is Distribution
        self.distribution = distObject
        self.dimension = distObject.dimension
        self.uniform0to1 = BatchRandomNumber(np.random.uniform, {'low':0.0, 'high':1.0})

    def acceptOrReject(self, proposedCoords):
        proposedDensity = self.distribution.density(proposedCoords)
        ratio = proposedDensity / self.currentDensity
        return (ratio > self.uniform0to1.draw()), proposedDensity
    
    def updateCoords(self, nextCoords, nextDensity):
        assert nextDensity < 1e+12
        self.currentCoords = nextCoords
        self.currentDensity = nextDensity
        
    def resetSample(self, initialCoords):
        if initialCoords is None:
            initialCoords = self.defaultStarter()
        self.monitor   = {'proposed': 0, 'accepted': 0}
        self.sampled = [initialCoords]
        self.currentCoords  = initialCoords
        self.currentDensity = self.distribution.density(initialCoords) + 1e-16

    def unitSample(self, length, initialCoords=None):
        self.resetSample(initialCoords)
        for i in range(0, length-1):
            nextCoords, nextDensity = self.proposeUntilAccepted()
            self.updateCoords(nextCoords, nextDensity)
            self.sampled.append(nextCoords)
        return self.sampled[:]

    def chainSample(self, length, burnIn, burnStep, initialCoords=None):
        assert burnIn >= 0
        assert burnStep >= 1
        assert length > (burnStep * 10)
        rawSample = self.unitSample(burnIn, initialCoords)
        retlist = []
        for i in range(0, burnStep+1):
            rawSample = self.unitSample(length, rawSample[-1])
            retlist += rawSample[burnStep::burnStep]
        return retlist[-length:]

    @abstractmethod
    def proposeUntilAccepted(self, maxAttempts=1000):
        pass

class GaussianMetropolis(Metropolis):
    '''
    Metropolis sampling from a distribution, using a Gaussian for transition proposals.
    '''
    def __init__(self, distObject, sigma):
        super(GaussianMetropolis, self).__init__(distObject)
        assert sigma > 1e-12
        self.deviation = sigma
        self.gaussian    = BatchRandomNumber(np.random.normal, {'loc':0.0, 'scale':self.deviation, 'size': self.dimension})
   
    def defaultStarter(self):
        return 10.0 * self.gaussian.draw()

    def proposeUntilAccepted(self, maxAttempts=1000):
        for i in range(0, maxAttempts):
            self.monitor['proposed'] += 1
            proposedCoords = np.add(self.currentCoords, self.gaussian.draw())
            accepted, proposedDensity = self.acceptOrReject(proposedCoords)
            if accepted:
                self.monitor['accepted'] += 1
                return proposedCoords, proposedDensity
        raise Exception('GaussianMetropolis stuck at local peak: {0}'.format(self.currentCoords))
    
    
        
