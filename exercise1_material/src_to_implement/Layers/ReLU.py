from Layers.Base import BaseLayer
import numpy as np
class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        
    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        # np.maximum: elementwise comparison and return max value
        return np.maximum(0, self.input_tensor)

    def backward(self, error_tensor):

        # if input_tensor <= 0: information gets lost -> here: return 0 instead return the error_tensor values
        return np.where(self.input_tensor > 0, error_tensor, 0) #element-wise