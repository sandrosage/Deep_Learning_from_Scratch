import copy

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self._phase = False


        @property
        def phase(self):
            return self._phase
        @phase.setter
        def phase(self, value):
            self._phase = value

    def forward(self):
        # get the batch from the data input layer
        input_tensor, self.label_tensor = self.data_layer.next()

        r_loss = 0

        # loop over all the hidden layers
        for layer in self.layers:
            layer.testing_phase = self._phase
            input_tensor = layer.forward(input_tensor)

            # add regularization loss to the data loss -> get regularization loss inside all layers and sum it up
            if hasattr(layer, 'optimizer'):
                if layer.optimizer:
                        if layer.optimizer.regularizer:
                            r_loss += layer.optimizer.regularizer.norm(layer.weights)

        # use last input_tensor as prediction_tensor and compute the loss regarding the gt-label_tensor
        output_tensor = self.loss_layer.forward(input_tensor, self.label_tensor) + r_loss
        return output_tensor
    
    def backward(self):
        # backpropagation: last layer -> first layer
        # first error_tensor of loss
        error_tensor = self.loss_layer.backward(self.label_tensor)

        # loop over again, BUT in reversed direction: last layer -> first layer
        for layer in reversed(self.layers):
            # within the layer.backward() the weights of the layers are updated using the optimizer
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        # check if layer is trainable
        if layer.trainable:
            #create a deep copy of the NeuralNetwork optimizer
            layer.initialize(self.weights_initializer, self.bias_initializer)
            layer.optimizer = copy.deepcopy(self.optimizer)
        # append the layer to the self.layers-list
        self.layers.append(layer)

    def train(self, iterations):
        self._phase = False
        # one forward/backward step for each iteration
        for i in range(iterations):
            loss = self.forward()

            #append the loss for each forward iteration
            self.loss.append(loss)
            
            #backward step with weights update
            self.backward()

    def test(self, input_tensor):
        self._phase = True

        # test: no backpropagation is needed only foward input_tensor
        for layer in self.layers:
            layer.testing_phase = self._phase
            input_tensor = layer.forward(input_tensor)
        # last input_tensor is prediction_tensor (for loss_layer)
        self.prediction_tensor = input_tensor
        return self.prediction_tensor