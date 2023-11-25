import numpy as np
from Layers.Base import BaseLayer
import copy
from Layers.Helpers import compute_bn_gradients

class BatchNormalization(BaseLayer):
    def __init__(self, channels) -> None:
        super().__init__()

        self.trainable = True
        self.channels = channels
        self.bias = 0
        self.weights = 0
        
        # epsilon: 1e-10 
        self.epsilon = np.finfo(float).eps

        # alpha value was given on the slides
        self.alpha = 0.8
        
        # mean
        self.mu = 0

        # variance
        self.sigma = 0 

        self.mu_tilted = 0
        self.sigma_tilted = 0
        self.x_tilted = 0
        self.input_tensor = None

        self._optimizer = None
        self.convolution = False

        self.initialize(None,None)

    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def initialize(self, weights_initializer,bias_initializer):

        # initialize weights with ones and biases with zero, since you don't want them to have an impact at the beginning of the training
        self.weights = np.ones(shape=self.channels)
        self.bias = np.zeros(shape=self.channels)
    
    # batch-normalization 
    def forward(self, input_tensor):
        self.input_tensor = copy.deepcopy(input_tensor)
        
        if len(np.shape(input_tensor)) > 2:
            input_tensor = self.reformat(copy.deepcopy(input_tensor))

            # helper variable to determine if in CNN or not
            self.convolution = True
        else:
            input_tensor = copy.deepcopy(input_tensor)


        # Testing
        if  self.testing_phase:


            self.x_tilted = (input_tensor - self.mu_tilted) / (np.sqrt(self.sigma_tilted**2 + self.epsilon))
            y_hat = self.weights * self.x_tilted + self.bias

        # Training
        else:
            
            self.mu = np.mean(input_tensor, axis=0)
            self.sigma = np.std(input_tensor, axis=0)
            
            # moving average for next test run
            self.mu_tilted = self.alpha * self.mu_tilted + (1 - self.alpha) * self.mu
            self.sigma_tilted = self.alpha * self.sigma_tilted + (1 - self.alpha) * self.sigma
            
            self.x_tilted = (input_tensor - self.mu) / (np.sqrt(self.sigma**2 + self.epsilon))
            y_hat = self.weights * self.x_tilted + self.bias
        
        if self.convolution:
            self.convolution = False
            return self.reformat(y_hat)
        else:
            return y_hat

    def backward(self, error_tensor):
        
        if len(error_tensor.shape) > 2:
            error_tensor = self.reformat(error_tensor)
            self.input_tensor = self.reformat(self.input_tensor)
            self.convolution = True

        #error w.r.t weights
        self.gradient_weights = np.sum(error_tensor * self.x_tilted, axis=0)

        #error w.r.t bias
        self.gradient_bias = np.sum(error_tensor, axis=0)

        #error w.r.t input
        Ex = compute_bn_gradients(error_tensor, self.input_tensor, self.weights, self.mu, self.sigma**2)

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.optimizer.calculate_update(self.bias, self.gradient_bias)
        
        if self.convolution:
            self.convolution = False
            return self.reformat(Ex)
        
        return Ex
    
    def reformat(self,tensor):
        # just use the formulas on the slides 
        if len(np.shape(tensor)) > 3:
            self.B,self.H,self.M,self.N = tensor.shape[0],tensor.shape[1],tensor.shape[2],tensor.shape[3]
            tensor = tensor.reshape(self.B,self.H, self.M*self.N)
            tensor = tensor.transpose((0,2,1))      
            tensor = tensor.reshape(self.B*self.M*self.N, self.H)
            return tensor
        if len(np.shape(tensor)) == 2:
            tensor = tensor.reshape(self.B, self.M*self.N,self.H)
            tensor = tensor.transpose((0,2,1))
            tensor = tensor.reshape(self.B, self.H, self.M, self.N)
            return tensor
        return tensor