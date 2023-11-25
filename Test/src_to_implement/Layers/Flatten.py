import numpy as np
from Layers.Base import BaseLayer

class Flatten(BaseLayer):
    def __init__(self) -> None:
        super().__init__()
        self.org_dims = None

    def forward(self, input_tensor):
        self.org_dims = input_tensor.shape
        return input_tensor.reshape(self.org_dims[0], -1)
    
    def backward(self, error_tensor):
        return error_tensor.reshape(self.org_dims)


