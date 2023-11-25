from Layers.FullyConnected import FullyConnected
from Layers.ReLU import ReLU
from Layers.SoftMax import SoftMax
import numpy as np
from Optimization.Loss import CrossEntropyLoss
from Optimization.Optimizers import Sgd
from Layers import Helpers
from NeuralNetwork import NeuralNetwork


net = NeuralNetwork(Sgd(1))
categories = 3
input_size = 4
net.data_layer = Helpers.IrisData(50)
net.loss_layer = CrossEntropyLoss()
fcl_1 = FullyConnected(input_size, categories)
net.append_layer(fcl_1)
net.append_layer(ReLU())
fcl_2 = FullyConnected(categories, categories)
net.append_layer(fcl_2)
net.append_layer(SoftMax())

out = net.forward()
out2 = net.forward()