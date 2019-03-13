class BatchRandomNumber(object):
    '''
    Batch RNG in NumPy for running time reduction.
    '''
    def __init__(self, generatorFunction, generatorArgs, batchSize=1024):
        '''
        Store the generator function and modify its arguments for batch generation. 
        Initialize a batch to hold RNG values.
        '''
        assert batchSize >= 32
        assert generatorFunction(**generatorArgs) is not None
        self.__batchSize = int(batchSize)
        self.__generator = generatorFunction
        self.__funcArgs = dict(generatorArgs)
        self.__batchified = False
        self.__batchify()
        self.resetBatch()
        
    def __batchify(self):
        '''
        Modify the size parameter in random number generation to accommodate batches.
        '''
        if self.__batchified:
            raise Exception('BatchRandomNumbers: __batchify() should not be called more than once.')
        if not 'size' in self.__funcArgs.keys():
            self.__funcArgs['size'] = self.__batchSize
        else:
            sizePerPiece = self.__funcArgs['size']
            if isinstance(sizePerPiece, int):
                self.__funcArgs['size'] = (self.__batchSize, sizePerPiece)
            elif isinstance(sizePerPiece, tuple):
                self.__funcArgs['size'] = (self.__batchSize,) + sizePerPiece
            else:
                raise Exception('BatchRandomNumbers: invalid size for random numbers, expected int or tuple, got {0} which is {1}.'.format(sizePerPiece, type(sizePerPiece)))
        self.__batchified = True
        
    def draw(self):
        '''
        Draw a random number from the pre-computed batch.
        '''
        if self.__batchCount < 1:
            self.resetBatch()
        self.__batchCount -= 1
        return self.__batch[self.__batchCount]
        
    def resetBatch(self):
        '''
        Pre-compute another batch and replace the current one.
        '''
        self.__batch = self.__generator(**self.__funcArgs)
        self.__batchCount = self.__batchSize
