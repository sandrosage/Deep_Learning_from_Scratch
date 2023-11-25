from Layers.Base import BaseLayer
import numpy as np

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size

        # weight matrix with randomly uniformly drawn weights between 0 and 1: expand dimension + 1 for bias weights
        self.weights = np.random.rand(self.input_size + 1, self.output_size) # W'
        self._optimizer = None

        # gradient matrix of weights: just create it with zeros of equal size like weight matrix
        self.gradient_weights = np.zeros_like(self.weights)
    
   
    # getter and setter method for property optimizer
    def get_optimizer(self):
        return self._optimizer
    
    def set_optimizer(self, optimizer):
        self._optimizer = optimizer

    optimizer = property(get_optimizer, set_optimizer)



    def return_weights(self):
        return self.weights
    

    def forward(self, input_tensor):
        if input_tensor.ndim == 1:
            # np.ones for the including the bias
            self.input_tensor = np.concatenate((input_tensor, np.ones((1)))) # X'
            # self.batch_size = 1 
        
        else:
            # .shape[0]: rows of matrix .shape[1]: columns of matrix
            # batch_size: represents the number of inputs processed simultaneously
            self.batch_size = input_tensor.shape[0]

            # concatenate: add bias as being the last column with axis=1 -> on y-axis
            self.input_tensor = np.concatenate((input_tensor, np.ones([self.batch_size, 1])), axis=1)

        # matrix product: @-operator
        return np.matmul(self.input_tensor, self.weights) # Y'
    
    def backward(self, error_tensor):
        # return error_tensor for the previous layer: opposite direction as forward process
        # E_n-1 = E_n*W_T
        gradient = np.matmul(error_tensor, self.weights.T)
        # if self.input_tensor.ndim == 1:
        #     # input_tensor = np.expand_dims(self.input_tensor, axis=0)
        #     # self.gradient_weights = np.matmul(input_tensor.T, np.expand_dims(error_tensor, axis=0))
        #     self.gradient_weights = np.matmul(error_tensor,self.input_tensor.T)
        # else:
        #X_t * E_n
        self.gradient_weights = np.matmul(self.input_tensor.T, error_tensor)

        # updating the weights
        # check if optimizer is set
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)

        # if error_tensor.ndim == 1:
        #     print("hello1")
        #     return gradient[:-1]

        #removing the bias column from the input.
        return gradient[:,:-1]

