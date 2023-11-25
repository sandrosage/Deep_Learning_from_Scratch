from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid
import numpy as np

class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_state = np.zeros(self.hidden_size)
        self._memorize = False
        self._optimizer = None

        # for the steps inside the Elman RNN cell we need two FCNN: 1 for incoming hidden state + input and 1 for calculating the output
        # first set the input dimensions for the firt FCNN
        # because they will be concatenated together afterwards, we can simply add the sizes
        self.input_size_FCNN_1 = self.input_size + self.hidden_size  

        # now create the two FCNN for each RNN-cell:
        # the output_size for the first FCNN will be the hidden_size, bc. it denotes the dimensions of the hidden states
        self.FCNN_1 = FullyConnected(self.input_size_FCNN_1, self.hidden_size)

        # second FCNN which calculates the output:
        # inputs are now the hidden states (h_t)
        self.FCNN_2 = FullyConnected(self.hidden_size, self.output_size)

        # 2 activations are needed: TanH and Sigmoid
        self.sigmoid = Sigmoid()
        self.tanH = TanH()

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, memorize):
        self._memorize = memorize

    @property
    def weights(self):   
        return self.FCNN_1.weights

    @weights.setter
    def weights(self, weights):
        self.FCNN_1.weights = weights

    @property
    def gradient_weights(self):
        return self.grad_FCNN_1
    
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer  

    def forward(self, input_tensor):
        # get the batch_size of the input 
        time_sequence = input_tensor.shape[0]  
        
        # create output_tensor w.r.t batch_size and output dimension 
        self.output_tensor = np.zeros((time_sequence, self.output_size)) 

        # set hidden state for this iteration to zero, otherwise restore the hidden state from the last iteration
        if not self._memorize:
            self.hidden_state = np.zeros(self.hidden_size)

         
        self.FCNN_1_memory = []
        self.FCNN_2_memory = []
        self.tanH_memory = []
        self.sigmoid_memory =[]

        for t in range(time_sequence):
           
            hidden_state_reshaped = self.hidden_state.reshape(self.hidden_size, 1)  
            input_tensor_reshaped = input_tensor[t].reshape(input_tensor.shape[1], 1)  
            
            # 1.Step: Concatenate input & hidden state -> x_tilted

            x_tilted = (np.concatenate((hidden_state_reshaped, input_tensor_reshaped))).T  

            #2.Step: FCNN_1 w.r.t. x_tilted for calculating hidden state
            input_for_tanh = self.FCNN_1.forward(x_tilted)  

            #3.Step: TanH for calculating the hidden state h_t
            self.hidden_state = self.tanH.forward(input_for_tanh)   

            #4.Step: FCNN_2 w.r.t hidden state for calculating output 
            forward_FC_2 = self.FCNN_2.forward(self.hidden_state)  

            #5.Step: Sigmoid for calculating the output y_t
            self.output_tensor[t] = self.sigmoid.forward(forward_FC_2)  

            # store the input of the forward pass -> needed in backward process
            self.FCNN_1_memory.append(self.FCNN_1.input_tensor)
            self.tanH_memory.append(self.tanH.activation)
            self.FCNN_2_memory.append(self.FCNN_2.input_tensor)
            self.sigmoid_memory.append(self.sigmoid.activation)
            

        return self.output_tensor 

    def backward(self, error_tensor):
        #placeholder
        output_tensor = np.zeros((error_tensor.shape[0], self.input_size))  

        #accumulate gradients
        self.grad_FCNN_1 = 0.0
        self.grad_FCNN_2 = 0.0

        gradient_hidden_state = np.zeros(self.hidden_size)  

        time_sequence = error_tensor.shape[0] 
        
        for t in reversed(range(time_sequence)):
            # 1.Step: Sigmoid backward step
            self.sigmoid.activation = self.sigmoid_memory[t]
            backward_sigmoid = self.sigmoid.backward(error_tensor[t])  
            
            # 2.Step: FCNN_2 backward step
            self.FCNN_2.input_tensor = self.FCNN_2_memory[t]
            backward_FCNN_2 = self.FCNN_2.backward(backward_sigmoid)  
            self.grad_FCNN_2 = self.grad_FCNN_2 + self.FCNN_2.gradient_weights  

            second_node = backward_FCNN_2 + gradient_hidden_state 

            # 4.Step: tanH bachward step
            self.tanH.activation = self.tanH_memory[t]
            backward_tanH = self.tanH.backward(second_node) 

            # 5.Step: fully FCNN_1 backward step
            self.FCNN_1.input_tensor = self.FCNN_1_memory[t]
            backward_FCNN_1 = self.FCNN_1.backward(backward_tanH) 
            self.grad_FCNN_1 = self.grad_FCNN_1 + self.FCNN_1.gradient_weights

            #remove reshape from forward
            output_tensor[t] = np.squeeze(backward_FCNN_1.T[self.hidden_size::])
            gradient_hidden_state = np.squeeze(backward_FCNN_1.T[0:self.hidden_size])

        
        self.gradient_weights_out = np.asarray(self.grad_FCNN_2)
        self.gradient_weights_hidden = np.asarray(self.grad_FCNN_1)
        if self._optimizer is not None:
            self.FCNN_1.weights = self.optimizer.calculate_update(self.FCNN_1.weights, self.gradient_weights_hidden)
            self.FCNN_2.weights = self.optimizer.calculate_update(self.FCNN_2.weights, self.gradient_weights_out)

        return output_tensor

    def initialize(self, weights_initializer, bias_initializer):
        # initialize both trainable weights/biases inside the FCNNs
        self.FCNN_1.initialize(weights_initializer, bias_initializer)
        self.FCNN_2.initialize(weights_initializer, bias_initializer)