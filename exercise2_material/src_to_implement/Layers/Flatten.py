from Layers.Base import BaseLayer
import numpy as np

class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.input_tensor_dim = input_tensor.shape
        return input_tensor.reshape(input_tensor.shape[0], -1)
    
    def backward(self, error_tensor):
        return error_tensor.reshape(self.input_tensor_dim)