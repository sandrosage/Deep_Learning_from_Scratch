import numpy as np
from Layers.Base import BaseLayer


class ReLU(BaseLayer):
    def __init__(self) -> None:
        super().__init__()
        self.input_tensor = None
    
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        #set num < 0 to 0 
        neg_nums = (input_tensor < 0)
        input_tensor[neg_nums] = 0

        return input_tensor
        

    def backward(self, error_tensor):
       grad = self.input_tensor > 0
       return error_tensor * grad


