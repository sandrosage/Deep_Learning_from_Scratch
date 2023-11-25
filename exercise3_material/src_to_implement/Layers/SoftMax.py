from Layers.Base import BaseLayer
import numpy as np

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        # mitigate high activations of input_tensor
        # max is used to find highest value per input
        # for numeric stability: hat(x_k) = x_k - max(x)
        hat_input_tensor = input_tensor - np.max(input_tensor,axis=1).reshape(-1,1)
        #apply softmax to every input
        numerator = np.exp(hat_input_tensor)
        # this is the same as exp(x_k) / sum_(j=1;k)( exp(x_j) )
        res = numerator / np.sum(numerator,axis=1).reshape(-1,1)
        self.probs = res
        return self.probs
    
    def backward(self, error_tensor):
    
        term_1 = (error_tensor * self.probs)
        
        sum_1 = np.sum(term_1,axis=1).reshape(-1,1)
        
        return self.probs * (error_tensor - sum_1)