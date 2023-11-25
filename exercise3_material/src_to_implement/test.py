import numpy as np
from Layers import Dropout, Helpers

class L2Loss:

    def __init__(self):
        self.input_tensor = None

    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        return np.sum(np.square(input_tensor - label_tensor))

    def backward(self, label_tensor):
        return 2*np.subtract(self.input_tensor, label_tensor)
    
batch_size = 10
input_size = 10
input_tensor = np.ones((batch_size, input_size))
label_tensor = np.zeros([batch_size, input_size])
for i in range(batch_size):
    label_tensor[i, np.random.randint(0, input_size)] = 1

print(label_tensor)
layers = list()
layers.append(Dropout.Dropout(0.5))
layers.append(L2Loss())
# difference = Helpers.gradient_check(layers, input_tensor, label_tensor, seed=1337)
difference = Helpers.gradient_check(layers, input_tensor, label_tensor)
# assertLessEqual(np.sum(difference), 1e-5)
    