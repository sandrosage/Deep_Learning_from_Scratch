import numpy as np
import copy

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        
        self.optimizer = optimizer

        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        
        self.loss = list()
        
        self.layers= list()
        
        self.data_layer = None
        
        self.loss_layer = None
        
        self.in_label = None


    def forward(self):
        in_data,in_label = self.data_layer.next() 
        self.in_label = in_label

        output_tensor = self.layers[0].forward(in_data)
        
        for layer in self.layers[1:]:
            output_tensor = layer.forward(output_tensor)

        loss = (self.loss_layer.forward(output_tensor,in_label))
        
        return loss
            

    def backward(self):
        loss = self.loss_layer.backward(self.in_label)
        layers = self.layers[::-1]
        output_tensor = layers[0].backward(loss)
        for layer in layers[1:]:
            output_tensor = layer.backward(output_tensor)

    def append_layer(self,layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            
            layer.initialize(self.weights_initializer, self.bias_initializer)

        self.layers.append(layer)

    def train(self,iterations):
        for iter in range(iterations+1):
            #for _ in range(self.data_layer.split // self.data_layer.batch_size):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self,input_tensor):
        output_tensor = self.layers[0].forward(input_tensor)
        for layer in self.layers[1:]:
            output_tensor = layer.forward(output_tensor)
        
        return output_tensor 
