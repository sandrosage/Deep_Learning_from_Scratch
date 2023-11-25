import numpy as np
from scipy import signal
from Layers.Base import BaseLayer
import Layers.Flatten
#import Helpers


class Pooling(BaseLayer):
    # ONLY FOR 2D OUTPUT - INPUT
    def __init__(self, stride_shape, pooling_shape) -> None:
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.input_tensor = None
        self.index = None
        self.max_vals = list()


    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        
        if len(input_tensor.shape) >= 3:
            samples = input_tensor.shape[0]
            channels = input_tensor.shape[1]
            y = input_tensor.shape[2]
            x = input_tensor.shape[3]
            output_y = int( (y - self.pooling_shape[0]) / self.stride_shape[0]) + 1
            output_x = int( (x - self.pooling_shape[1]) / self.stride_shape[1]) + 1
            output_shape = (samples,channels, output_y, output_x)

            output = np.zeros(output_shape)

            #self.weights = np.random.uniform(0, 1, (self.pooling_shape))
            #for every sample
            for s in range(input_tensor.shape[0]):
                #for every channel
                for c in range(input_tensor.shape[1]):
                    for w in range(output_y):
                        for h in range(output_x):
                            #find indices for elements from input_tensor inside kernel
                            v_start = w * self.stride_shape[0]
                            v_end = w * self.stride_shape[0] + self.pooling_shape[0]
                            h_start = h * self.stride_shape[1] 
                            h_end = h * self.stride_shape[1] + self.pooling_shape[1]
                            #get elements
                            slice_ = input_tensor[s,c,v_start:v_end,h_start:h_end]
                            #find max
                            output[s,c, w, h] = np.max(slice_)
                          
              
            return output
        
        else:
            return input_tensor

    def backward(self, error_tensor):
        sY, sX = self.stride_shape
        pY, pX = self.pooling_shape
        m,n_C, n_Y, n_X = error_tensor.shape

        # Initializing with zeros
        error = np.zeros(self.input_tensor.shape)

        for i in range(m):
            input1 = self.input_tensor[i]
            for y in range(n_Y):
                for x in range(n_X):
                    for c in range(n_C):
                        # Find the borders
                        y_start = y * sY
                        y_end = y_start + pY
                        x_start = x * sX
                        x_end = x_start + pX

                        input_slice = input1[c,y_start:y_end, x_start:x_end]
                        # max pooling for mask
                        mask = input_slice == np.max(input_slice)
                        # calculating the derivative
                        error[i,c, y_start:y_end, x_start:x_end] += np.multiply(mask,error_tensor[i,c, y, x])
        return error


class L2Loss:

    def __init__(self):
        self.input_tensor = None

    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        return np.sum(np.square(input_tensor - label_tensor))

    def backward(self, label_tensor):
        return 2*np.subtract(self.input_tensor, label_tensor)


