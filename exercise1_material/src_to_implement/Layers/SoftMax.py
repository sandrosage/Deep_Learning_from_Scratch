from Layers.Base import BaseLayer
import numpy as np

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        
        # for numeric stability: hat(x_k) = x_k - max(x)
        hat_input_tensor = input_tensor - np.max(input_tensor)

        # devide the whole calculation into small steps and then compose them
        nom = np.exp(hat_input_tensor)
        den = np.sum(nom, axis=1)
        print(np.expand_dims(den, axis=1))
        self.probs = np.divide(nom, np.expand_dims(den, axis=1)) # np.expand_dims() because it should be a column vector.
        return self.probs
    
    def backward(self, error_tensor):
        sum = np.sum(error_tensor * self.probs, axis=1)
        sum = np.expand_dims(sum, axis=1) # np.expand_dims() because it should be a column vector.
        return (self.probs*(error_tensor-sum))