import numpy as np

class Constant:
    def __init__(self, default = 0.1) -> None:
        self.weights_shape = None
        self.fan_in = None
        self.fan_out = None
        self.default = default
    
    def initialize(self, weights_shape, fan_in, fan_out):
        self.weights_shape = weights_shape
        self.fan_in = fan_in
        self.fan_out = fan_out
        
        weights = np.full(weights_shape,self.default)
        
        return weights

class UniformRandom:
    def __init__(self) -> None:
        self.weights_shape = None
        self.fan_in = None
        self.fan_out = None
    
    def initialize(self, weights_shape, fan_in, fan_out):
        self.weights_shape = weights_shape
        self.fan_in = fan_in
        self.fan_out = fan_out
        weights = np.random.uniform(0,1,(weights_shape))

        return weights

class Xavier:
    def __init__(self) -> None:
        self.weights_shape = None
        self.fan_in = None
        self.fan_out = None
    
    def initialize(self, weights_shape, fan_in, fan_out):
        self.weights_shape = weights_shape
        self.fan_in = fan_in
        self.fan_out = fan_out  

        sigma = np.sqrt(2/(fan_out+fan_in))
        weights = np.random.normal(0,sigma,weights_shape)
        
        return weights

class He:
    def __init__(self) -> None:
        self.weights_shape = None
        self.fan_in = None
        self.fan_out = None
    
    def initialize(self, weights_shape, fan_in, fan_out):
        self.weights_shape = weights_shape
        self.fan_in = fan_in
        self.fan_out = fan_out  

        sigma = np.sqrt(2/(fan_in))
        weights = np.random.normal(0,sigma,weights_shape)
        
        return weights    

