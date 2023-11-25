from Layers.Base import BaseLayer
import numpy as np


class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape) -> None:
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.input_tensor = None
        self.index = None
        self.max_vals = list()


    def forward(self, input_tensor):
        
        # store it for backward step
        self.input_tensor = input_tensor
        

        if len(input_tensor.shape) >= 3:
            samples, channels, y, x = input_tensor.shape

            # determine the output_shape after pooling
            output_y = int( (y - self.pooling_shape[0]) / self.stride_shape[0]) + 1
            output_x = int( (x - self.pooling_shape[1]) / self.stride_shape[1]) + 1
            output_shape = (samples,channels, output_y, output_x)

            output = np.zeros(output_shape)

            #for every sample
            for s in range(samples):
                #for every channel
                for c in range(channels):
                    for width in range(output_y):
                        for height in range(output_x):
                            #find indices for elements from input_tensor inside kernel
                            v_start = width * self.stride_shape[0]
                            v_end = width * self.stride_shape[0] + self.pooling_shape[0]
                            h_start = height * self.stride_shape[1] 
                            h_end = height * self.stride_shape[1] + self.pooling_shape[1]
                            #get elements
                            slice = input_tensor[s,c,v_start:v_end,h_start:h_end]
                            
                            # use MaxPooling: np.max() -> return max. element in matrix
                            output[s,c, width, height] = np.max(slice)
                           
            return output
        

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
                        # accumulate it to not loose the first one += not assign it
                        error[i,c, y_start:y_end, x_start:x_end] += np.multiply(mask,error_tensor[i,c, y, x])
        return error