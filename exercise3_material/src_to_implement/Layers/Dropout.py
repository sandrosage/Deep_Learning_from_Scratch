from Layers.Base import BaseLayer
import numpy as np
import copy

class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()

        self.probability = probability
        self.binary_mask = None
        self.idx = None

    def forward(self, input_tensor):
        input_tensor = copy.deepcopy(input_tensor)
        # in test time, no dropout!
        if self.testing_phase == True:
            return input_tensor

        self.binary_mask = np.random.rand(input_tensor.shape[0], input_tensor.shape[1]) < self.probability
        output_tensor = input_tensor * self.binary_mask
        #find where the 0s are
        self.idx = np.where(output_tensor == 0)
        # inverted dropout
        output_tensor /= self.probability
        return output_tensor

    def backward(self, error_tensor):
        error_tensor = copy.deepcopy(error_tensor)
        error_tensor[self.idx] = 0
        return (1/self.probability) * error_tensor