from Layers.Base import BaseLayer
import numpy as np


class SoftMax(BaseLayer):
    def __init__(self) -> None:
        super().__init__()
        self.input_res = None

    def forward(self, input_tensor):
        #mitigate high activations of input_tensor
        #max is used to find highest value per input
        input_tensor = input_tensor - np.max(input_tensor,axis=1).reshape(-1,1)
        #apply softmax to every input
        numerator = np.exp(input_tensor)
        # this is the same as exp(x_k) / sum_(j=1;k)( exp(x_j) )
        res = numerator / np.sum(numerator,axis=1).reshape(-1,1)
        self.input_res = res
        return res

    def backward(self,error_tensor):
        term1 = (error_tensor * self.input_res)
        
        sum1 = np.sum(term1,axis=1).reshape(-1,1)
        
        res = error_tensor - sum1
        res = self.input_res * res
        return res


